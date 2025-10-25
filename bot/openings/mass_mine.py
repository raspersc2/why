from random import choice
from typing import Callable

from sc2.ids.ability_id import AbilityId

from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import DropCargo, KeepUnitSafe, PathUnitToTarget
from ares.behaviors.macro import BuildStructure
from ares.consts import TOWNHALL_TYPES
from cython_extensions import (
    cy_distance_to_squared,
    cy_in_pathing_grid_ma,
    cy_sorted_by_distance_to,
    cy_towards,
    cy_center,
    cy_unit_pending,
)
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.mine_combat import MineCombat
from bot.openings.battle_cruiser_rush import BattleCruiserRush
from src.ares.consts import UnitRole

from bot.combat.base_combat import BaseCombat
from bot.combat.medivac_mine_drops import MedivacMineDrops
from bot.openings.opening_base import OpeningBase
from bot.openings.reapers import Reapers

ARMY_TYPES: set[UnitTypeId] = {
    UnitTypeId.BATTLECRUISER,
    UnitTypeId.MEDIVAC,
    UnitTypeId.WIDOWMINE,
}
DROP_ROLES: set[UnitRole] = {
    UnitRole.DROP_SHIP,
    UnitRole.DROP_UNITS_TO_LOAD,
    UnitRole.DROP_UNITS_ATTACKING,
}
MINE_TYPES: set[UnitTypeId] = {UnitTypeId.WIDOWMINE, UnitTypeId.WIDOWMINEBURROWED}
STEAL_FROM_ROLES: set[UnitRole] = {UnitRole.ATTACKING, UnitRole.DEFENDING}


class MassMine(OpeningBase):
    _battle_cruisers: OpeningBase
    _reapers: OpeningBase
    _mine_combat: BaseCombat
    _mine_drops: BaseCombat
    _natural_position: Point2
    _main_ramp_pos: Point2

    def __init__(self):
        super().__init__()
        # {med_tag: {"mine_tags": {tag1, tag2, ...}, "target": Point2(...)}}
        self._medivac_tag_to_mine_tracker: dict[int, dict] = dict()
        self.MIN_HEALTH_MEDIVAC_PERC: float = 0.3
        # Track which bases have defense mines: {townhall_tag: [mine_tag1, mine_tag2]}
        self._base_defense_mines: dict[int, list[int]] = dict()

        self._defensive_mine_positions: dict[Point2, Point2] = dict()
        self._main_ramp_mines: list[int] = []
        self._switched_to_bcs: bool = False
        self.defensive: bool = True

    @property
    def army_comp(self) -> dict:
        own_army_dict = self.ai.mediator.get_own_army_dict
        if len(own_army_dict[UnitTypeId.MEDIVAC]) < 4:
            return {
                UnitTypeId.WIDOWMINE: {"proportion": 0.8, "priority": 1},
                UnitTypeId.MEDIVAC: {"proportion": 0.2, "priority": 0},
            }
        else:
            return {
                UnitTypeId.WIDOWMINE: {"proportion": 1.0, "priority": 0},
            }

    @property
    def upgrade_list(self) -> list[UpgradeId]:
        if len(self.ai.townhalls) < 2:
            return [UpgradeId.DRILLCLAWS]
        return [UpgradeId.DRILLCLAWS, UpgradeId.HISECAUTOTRACKING]

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._battle_cruisers = BattleCruiserRush()
        await self._battle_cruisers.on_start(ai)
        self._reapers = Reapers()
        await self._reapers.on_start(ai)

        self._mine_combat = MineCombat(ai, ai.config, ai.mediator)
        self._mine_drops = MedivacMineDrops(ai, ai.config, ai.mediator)
        self._natural_position: Point2 = Point2(
            cy_towards(self.ai.mediator.get_own_nat, self.ai.game_info.map_center, 7.0)
        )
        self._main_ramp_pos = Point2(
            cy_towards(
                self.ai.main_base_ramp.top_center,
                self.ai.main_base_ramp.bottom_center,
                2.5,
            )
        )

        # cache defensive mine positions so we don't have to
        # keep iterating over mfs
        for el in self.ai.expansion_locations_list:
            mineral_fields: list[Unit] = [
                mf
                for mf in self.ai.mineral_field
                if cy_distance_to_squared(mf.position, el) < 100.0
            ]
            if not mineral_fields:
                continue
            position: Point2 = Point2(cy_towards(cy_center(mineral_fields), el, 2.0))
            self._defensive_mine_positions[el] = position

    async def on_step(self) -> None:
        if self._switched_to_bcs:
            await self._battle_cruisers.on_step()
        elif self.ai.build_order_runner.build_completed:
            self._macro()

        if not self._switched_to_bcs and (
            self.ai.supply_used >= 167
            or (
                self.ai.time > 450.0
                and self.ai.get_total_supply(self.ai.mediator.get_cached_enemy_army) < 8
            )
        ):
            self._switched_to_bcs = True
            await self.ai.chat_send(f"Tag: {self.ai.time_formatted}_switched_to_bcs")

        await self._reapers.on_step()

        if (
            not self.ai.mediator.get_did_enemy_rush
            and self.ai.time > 300.0
            and not self.ai.mediator.get_main_ground_threats_near_townhall
        ):
            self.defensive = False

        # leave a couple mines on the main ramp for defense
        self._handle_main_ramp_mines()
        # assign a couple mines to go in each base for defense
        self._assign_base_defense_mines()
        # aggressive drops
        self._handle_drops()
        # handle mines that have lost their medivac
        self._cleanup_orphaned_drop_mines()

        # handle all left over mines
        main_force: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.ATTACKING
        )
        _target: Point2 = (
            self._natural_position if self.ai.time < 300.0 else self.attack_target
        )
        self._mine_combat.execute(
            main_force(MINE_TYPES), target=_target, stay_burrowed=self.defensive
        )

        # handle medivacs not currently dropping
        air_grid = self.ai.mediator.get_air_grid
        for medivac in main_force(UnitTypeId.MEDIVAC):
            medivac_maneuver: CombatManeuver = CombatManeuver()
            medivac_maneuver.add(KeepUnitSafe(medivac, grid=air_grid))
            if medivac.has_cargo and cy_in_pathing_grid_ma(air_grid, medivac.position):
                medivac_maneuver.add(DropCargo(medivac, medivac.position))
            medivac_maneuver.add(
                PathUnitToTarget(medivac, air_grid, self._natural_position)
            )
            self.ai.register_behavior(medivac_maneuver)

        # send base defence mines to their positions
        for th_tag, mine_tags in self._base_defense_mines.items():
            th: Unit | None = self.ai.unit_tag_dict.get(th_tag, None)
            if not th:
                continue

            position: Point2 = self._defensive_mine_positions[th.position]
            for mine_tag in mine_tags:
                mine: Unit | None = self.ai.unit_tag_dict.get(mine_tag, None)
                if not mine or mine.is_burrowed:
                    continue

                if cy_distance_to_squared(mine.position, position) < 3.0:
                    mine(AbilityId.BURROWDOWN_WIDOWMINE)
                else:
                    mine.move(position)

    def _handle_drops(self) -> None:
        self._execute_drops()
        if not self.ai.mediator.get_main_ground_threats_near_townhall:
            self._assign_mine_drops()
        self._unassign_mine_drops(UnitRole.ATTACKING)

    def _assign_base_defense_mines(self) -> None:
        """Assign 2 widow mines to defend each base mineral line, spaced apart."""

        for townhall in self.ai.townhalls:
            th_tag: int = townhall.tag
            if townhall.position not in self._defensive_mine_positions:
                continue

            # Clean up dead mines from tracking
            if th_tag in self._base_defense_mines:
                self._base_defense_mines[th_tag] = [
                    tag
                    for tag in self._base_defense_mines[th_tag]
                    if tag in self.ai.unit_tag_dict
                ]
            else:
                self._base_defense_mines[th_tag] = []

            # Check how many defense mines this base has
            current_count: int = len(self._base_defense_mines[th_tag])

            if current_count >= 2:
                continue

            # Need more defense mines
            needed: int = 2 - current_count

            # Get available mines from STEAL_FROM_ROLES
            available_units: Units = self.ai.mediator.get_units_from_roles(
                roles=STEAL_FROM_ROLES
            )
            available_mines: list[Unit] = [
                u for u in available_units if u.type_id in MINE_TYPES
            ]

            if not available_mines:
                continue

            # Get closest mines to this townhall
            closest_mines: list[Unit] = cy_sorted_by_distance_to(
                available_mines, townhall.position
            )[:needed]

            # Assign each mine to defense
            for i, mine in enumerate(closest_mines):
                self.ai.mediator.assign_role(tag=mine.tag, role=UnitRole.BASE_DEFENDER)
                self._base_defense_mines[th_tag].append(mine.tag)
                if mine.is_burrowed:
                    mine(AbilityId.BURROWUP_WIDOWMINE)

    def _cleanup_orphaned_drop_mines(self) -> None:
        """Reassign any DROP_UNITS_TO_LOAD mines not in tracker back to ATTACKING."""
        # Get all mine tags that are currently in active drops
        tracked_mine_tags: set[int] = set()
        for tracker_info in self._medivac_tag_to_mine_tracker.values():
            tracked_mine_tags.update(tracker_info["mine_tags"])

        # Get all units with DROP_UNITS_TO_LOAD role
        units_to_load: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.DROP_UNITS_TO_LOAD
        )

        # Reassign any that aren't in the tracker
        for unit in units_to_load:
            if unit.type_id in MINE_TYPES and unit.tag not in tracked_mine_tags:
                self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id in ARMY_TYPES:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    def _execute_drops(self) -> None:
        self._mine_drops.execute(
            self.ai.mediator.get_units_from_roles(roles=DROP_ROLES),
            medivac_tag_to_mine_tracker=self._medivac_tag_to_mine_tracker,
        )

    def _macro(self):
        freeflow_mode: bool = True
        if self.ai.time < 300.0 and not cy_unit_pending(self.ai, UnitTypeId.MEDIVAC):
            freeflow_mode = False
        production_controller_enabled: bool = True
        if self.ai.mediator.get_own_structures_dict[UnitTypeId.BARRACKSFLYING]:
            production_controller_enabled = False
        self._generic_macro_plan(
            self.army_comp,
            self.ai.start_location,
            self.upgrade_list,
            add_hellions=False,
            add_upgrades=True,
            can_expand=True,
            freeflow_mode=freeflow_mode,
            upgrade_to_pfs=True,
            production_controller_enabled=production_controller_enabled,
            num_one_base_workers=24,
        )

        # Build missile turrets at each base
        for townhall in self.ai.townhalls:
            if (
                not self.ai.can_afford(UnitTypeId.MISSILETURRET)
                or self.ai.structure_pending(UnitTypeId.MISSILETURRET)
                or len(
                    [
                        s
                        for s in self.ai.mediator.get_own_structures_dict[
                            UnitTypeId.ENGINEERINGBAY
                        ]
                        if s.is_ready
                    ]
                )
                == 0
            ):
                break

            location: Point2 = townhall.position
            existing_turrets: list[Unit] = [
                s
                for s in self.ai.structures
                if s.type_id == UnitTypeId.MISSILETURRET
                and cy_distance_to_squared(location, s.position) < 200.0
            ]
            if len(existing_turrets) < 2:
                self.ai.register_behavior(
                    BuildStructure(
                        location, UnitTypeId.MISSILETURRET, closest_to=location
                    )
                )

    def _assign_mine_drops(self) -> None:
        """Assign available medivacs and mines to new drop operations."""
        # Get units with ATTACKING or DEFENDING roles only
        available_units: Units = self.ai.mediator.get_units_from_roles(
            roles=STEAL_FROM_ROLES
        )
        if not available_units:
            return

        # Find available medivacs (not already assigned to a drop)
        medivacs: list[Unit] = [
            m
            for m in available_units
            if m.type_id == UnitTypeId.MEDIVAC
            and m.tag not in self._medivac_tag_to_mine_tracker
            and m.health_percentage >= 1.0
        ]
        if not medivacs:
            return

        # Find available widow mines (not already assigned to a drop)
        already_assigned_mine_tags: set[int] = set()
        for tracker in self._medivac_tag_to_mine_tracker.values():
            already_assigned_mine_tags.update(tracker["mine_tags"])

        mines: list[Unit] = [
            m
            for m in available_units
            if m.type_id in MINE_TYPES and m.tag not in already_assigned_mine_tags
        ]

        # Need at least 1 medivac and 4 mines
        if not mines or len(mines) < 4:
            return

        # Pick the first available medivac and closest 4 mines
        medivac: Unit = medivacs[0]
        selected_mines: list[Unit] = cy_sorted_by_distance_to(mines, medivac.position)[
            :4
        ]

        # Assign roles
        self.ai.mediator.assign_role(tag=medivac.tag, role=UnitRole.DROP_SHIP)
        for mine in selected_mines:
            self.ai.mediator.assign_role(tag=mine.tag, role=UnitRole.DROP_UNITS_TO_LOAD)
            if mine.is_burrowed:
                mine(AbilityId.BURROWUP_WIDOWMINE)

        # Calculate drop target
        target_base: Point2 = self.ai.enemy_start_locations[0]
        if self.ai.enemy_structures(TOWNHALL_TYPES):
            target_base = choice(
                [th.position for th in self.ai.enemy_structures(TOWNHALL_TYPES)]
            )

        drop_target: Point2 = Point2(
            cy_towards(target_base, self.ai.game_info.map_center, -4.0)
        )

        # Create tracker entry (store initial tags only - never modify)
        self._medivac_tag_to_mine_tracker[medivac.tag] = {
            "mine_tags": frozenset({mine.tag for mine in selected_mines}),
            "target": drop_target,
        }

    def _unassign_mine_drops(self, switch_to: UnitRole) -> None:
        """Clean up drop tracker entries when drops complete or fail."""
        if not self._medivac_tag_to_mine_tracker:
            return

        tags_to_remove: list[int] = []
        grid = self.ai.mediator.get_air_grid

        for med_tag, tracker in list(self._medivac_tag_to_mine_tracker.items()):
            medivac: Unit | None = self.ai.unit_tag_dict.get(med_tag, None)

            # Medivac died - remove tracker only
            if not medivac:
                tags_to_remove.append(med_tag)
                continue

            # Low health medivac - emergency drop and remove tracker
            if (
                medivac.health_percentage <= self.MIN_HEALTH_MEDIVAC_PERC
                and cy_in_pathing_grid_ma(grid, medivac.position)
            ):
                self.ai.mediator.assign_role(tag=medivac.tag, role=switch_to)
                self.ai.register_behavior(
                    DropCargo(unit=medivac, target=medivac.position)
                )
                tags_to_remove.append(med_tag)
                continue

            # Medivac has cargo - drop in progress, don't touch
            if medivac.has_cargo:
                continue

            # No cargo - check if any of the original mines still exist with DROP roles
            original_mine_tags: set[int] = set(tracker["mine_tags"])
            mines_with_drop_roles: list[Unit] = [
                u
                for u in self.ai.mediator.get_units_from_roles(roles=DROP_ROLES)
                if u.tag in original_mine_tags and u.type_id in MINE_TYPES
            ]

            # All original mines are gone or no longer have DROP roles - mission complete
            if not mines_with_drop_roles:
                self.ai.mediator.assign_role(tag=medivac.tag, role=switch_to)
                tags_to_remove.append(med_tag)

        # Clean up completed drops
        for tag in tags_to_remove:
            del self._medivac_tag_to_mine_tracker[tag]

    def _handle_main_ramp_mines(self):
        """Assign 2 widow mines to defend the main ramp."""
        # Cleanup dead mine tags
        self._main_ramp_mines = [
            tag for tag in self._main_ramp_mines if tag in self.ai.unit_tag_dict
        ]

        # Return if we already have enough mines
        if len(self._main_ramp_mines) >= 2:
            # Handle existing ramp mines
            for mine_tag in self._main_ramp_mines:
                mine = self.ai.unit_tag_dict.get(mine_tag)
                if not mine:
                    continue
                if cy_distance_to_squared(mine.position, self._main_ramp_pos) < 1.5:
                    mine(AbilityId.BURROWDOWN_WIDOWMINE)
                else:
                    if mine.is_burrowed:
                        mine(AbilityId.BURROWUP_WIDOWMINE)
                    else:
                        mine.move(self._main_ramp_pos)
            return

        # Need more mines
        needed = 2 - len(self._main_ramp_mines)
        available_units = self.ai.mediator.get_units_from_roles(roles=STEAL_FROM_ROLES)
        available_mines = [u for u in available_units if u.type_id in MINE_TYPES]

        if not available_mines:
            return

        # Get closest mines to ramp
        closest_mines = cy_sorted_by_distance_to(available_mines, self._main_ramp_pos)[
            :needed
        ]

        # Assign each mine to ramp defense
        for mine in closest_mines:
            self.ai.mediator.assign_role(tag=mine.tag, role=UnitRole.CONTROL_GROUP_ONE)
            self._main_ramp_mines.append(mine.tag)
            if mine.is_burrowed:
                mine(AbilityId.BURROWUP_WIDOWMINE)
