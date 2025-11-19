import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import AMove, KeepUnitSafe
from ares.behaviors.macro import SpawnController
from ares.cache import property_cache_once_per_frame
from ares.consts import ALL_STRUCTURES, UnitRole, UnitTreeQueryType
from cython_extensions import (
    cy_center,
    cy_closest_to,
    cy_in_pathing_grid_ma,
    cy_unit_pending,
)
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.battle_cruiser_combat import BattleCruiserCombat
from bot.combat.generic_drops import GenericDrops
from bot.combat.ground_range_combat import GroundRangeCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES
from bot.openings.bio import Bio
from bot.openings.opening_base import OpeningBase
from bot.openings.reapers import Reapers

ARMY_TYPES: set[UnitTypeId] = {UnitTypeId.MARINE, UnitTypeId.MEDIVAC, UnitTypeId.THOR}
DROP_ROLES: set[UnitRole] = {
    UnitRole.DROP_SHIP,
    UnitRole.DROP_UNITS_TO_LOAD,
    UnitRole.DROP_UNITS_ATTACKING,
}


class ThorDrop(OpeningBase):
    _worker_rush_activated: bool
    _battle_cruiser_combat: BaseCombat
    _ground_range_combat: BaseCombat
    _thor_drops: BaseCombat
    target_healing_pos: Point2
    _bio: OpeningBase
    _reapers: OpeningBase

    def __init__(self):
        super().__init__()
        self._attack_started: bool = False
        self._assign_repairers: bool = False
        self._medivac_to_thor: dict[int, dict] = dict()

    @property
    def army_comp(self) -> dict:
        return {
            UnitTypeId.THOR: {"proportion": 0.4, "priority": 0},
            UnitTypeId.MEDIVAC: {"proportion": 0.6, "priority": 1},
        }

    @property_cache_once_per_frame
    def healing_spot(self) -> Point2:
        return self.ai.mediator.find_closest_safe_spot(
            from_pos=self.target_healing_pos,
            grid=self.ai.mediator.get_ground_grid,
            radius=15.0,
        )

    @property
    def upgrade_list(self) -> list[UpgradeId]:
        return []

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id in ARMY_TYPES:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
        if unit.type_id == UnitTypeId.THOR:
            unit(AbilityId.MORPH_THORHIGHIMPACTMODE)

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._reapers = Reapers()
        await self._reapers.on_start(ai)
        self._bio = Bio()
        await self._bio.on_start(ai)
        self._battle_cruiser_combat = BattleCruiserCombat(ai, ai.config, ai.mediator)
        self._ground_range_combat = GroundRangeCombat(ai, ai.config, ai.mediator)
        self._thor_drops: BaseCombat = GenericDrops(ai, ai.config, ai.mediator)

        self.target_healing_pos = self.ai.game_info.map_center
        if path := self.ai.mediator.find_raw_path(
            start=self.ai.mediator.get_enemy_nat,
            target=self.ai.game_info.map_center,
            grid=self.ai.mediator.get_ground_grid,
            sensitivity=2,
        ):
            # make sure the path has some kind of length
            # the actual length is likely a lot longer
            if len(path) > 5:
                # take the second from the end point
                self.target_healing_pos = path[-2]

    async def on_step(self) -> None:
        await self._reapers.on_step()
        await self._micro()
        if not self.ai.build_order_runner.build_completed:
            return
        self._assign_drops()
        # update targets, check if need healing etc
        self._update_drop_info()
        self._macro()
        await self._micro()
        await self._handle_repair_crew()

    def _assign_drops(self) -> None:
        available_units: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.ATTACKING
        )
        medivacs: Units = available_units(UnitTypeId.MEDIVAC)
        thors: Units = available_units(UnitTypeId.THOR)
        # for simplicity assign one per frame
        if medivacs and thors:
            medivac: Unit = cy_closest_to(cy_center(thors), medivacs)
            self.ai.mediator.assign_role(tag=medivac.tag, role=UnitRole.DROP_SHIP)

            thor_tag: int = cy_closest_to(medivac.position, thors).tag
            self.ai.mediator.assign_role(
                tag=thor_tag,
                role=UnitRole.DROP_UNITS_TO_LOAD,
            )
            self._medivac_to_thor[medivac.tag] = {
                "tags": {thor_tag},
                "target": self.attack_target,
                "healing": False,
            }

    def _macro(self):
        self.ai.register_behavior(
            SpawnController(
                army_composition_dict={
                    UnitTypeId.MARINE: {"proportion": 1.0, "priority": 0}
                }
            )
        )
        pending_thors: bool = cy_unit_pending(self.ai, UnitTypeId.THOR)
        self._generic_macro_plan(
            self.army_comp,
            self.ai.start_location,
            self.upgrade_list,
            add_hellions=False,
            add_upgrades=pending_thors,
            can_expand=pending_thors,
            freeflow_mode=False,
            upgrade_to_pfs=False,
        )

    async def _micro(self) -> None:
        self._thor_drops.execute(
            self.ai.mediator.get_units_from_roles(roles=DROP_ROLES),
            medivac_tag_to_units_tracker=self._medivac_to_thor,
        )

        # handle left over units
        attack_target: Point2 = self.attack_target
        thors: Units = self.ai.mediator.get_own_army_dict[UnitTypeId.THOR]

        if not self._attack_started and thors:
            self._attack_started = True
        marine_target = (
            self.ai.main_base_ramp.top_center
            if not self._attack_started
            else attack_target
        )
        await self._bio.on_step(target=marine_target)

    def _update_drop_info(self):
        keys_to_remove: list[int] = []
        ground_grid: np.ndarray = self.ai.mediator.get_ground_grid
        for medivac_tag, tracker_info in self._medivac_to_thor.items():
            medivac: Unit | None = self.ai.unit_tag_dict.get(medivac_tag)
            thor: Unit | None = self.ai.unit_tag_dict.get(list(tracker_info["tags"])[0])
            if tracker_info["healing"]:
                self._medivac_to_thor[medivac_tag]["target"] = self.healing_spot
                if (
                    thor
                    and thor.health_percentage >= 1.0
                    and medivac
                    and medivac.health_percentage >= 1.0
                ):
                    self.ai.mediator.assign_role(
                        tag=thor.tag, role=UnitRole.DROP_UNITS_TO_LOAD
                    )
                    self._medivac_to_thor[medivac_tag][
                        "target"
                    ] = self.ai.enemy_start_locations[0]
                    self._medivac_to_thor[medivac_tag]["healing"] = False
                continue

            if medivac and medivac.has_cargo:
                if medivac.health_percentage < 0.25:
                    self._medivac_to_thor[medivac_tag]["healing"] = True
                    self._medivac_to_thor[medivac_tag]["target"] = self.healing_spot
                    self._assign_repairers = True
                    continue
                close_enemy_air: Units = self.ai.mediator.get_units_in_range(
                    start_points=[medivac.position],
                    distances=11.0,
                    query_tree=UnitTreeQueryType.EnemyFlying,
                )[0].filter(
                    lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES and u.is_visible
                )

                if close_enemy_air and cy_in_pathing_grid_ma(
                    ground_grid, medivac.position
                ):
                    self._medivac_to_thor[medivac_tag]["target"] = medivac.position
                    continue

                close_enemy_ground: Units = self.ai.mediator.get_units_in_range(
                    start_points=[medivac.position],
                    distances=7.0,
                    query_tree=UnitTreeQueryType.EnemyGround,
                )[0].filter(
                    lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES and u.is_visible
                )

                if close_enemy_ground and cy_in_pathing_grid_ma(
                    ground_grid, medivac.position
                ):
                    self._medivac_to_thor[medivac_tag]["target"] = medivac.position
                    continue

            if thor:
                if thor.health_percentage < 0.4:
                    self._medivac_to_thor[medivac_tag]["healing"] = True
                    self._medivac_to_thor[medivac_tag]["target"] = self.healing_spot
                    self.ai.mediator.assign_role(
                        tag=thor.tag, role=UnitRole.DROP_UNITS_TO_LOAD
                    )
                    self._assign_repairers = True
                    continue

                close_enemy_air: Units = self.ai.mediator.get_units_in_range(
                    start_points=[thor.position],
                    distances=11.0,
                    query_tree=UnitTreeQueryType.EnemyFlying,
                )[0].filter(
                    lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES and u.is_visible
                )

                close_enemy_ground: Units = self.ai.mediator.get_units_in_range(
                    start_points=[thor.position],
                    distances=7.0,
                    query_tree=UnitTreeQueryType.EnemyGround,
                )[0].filter(
                    lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES and u.is_visible
                )

                if not close_enemy_air and not close_enemy_ground:
                    close_units: Units = self.ai.mediator.get_units_in_range(
                        start_points=[thor.position],
                        distances=12.0,
                        query_tree=UnitTreeQueryType.AllEnemy,
                    )[0].filter(lambda u: u.type_id not in ALL_STRUCTURES)
                    if close_units:
                        self.ai.mediator.assign_role(
                            tag=thor.tag, role=UnitRole.DROP_UNITS_TO_LOAD
                        )
                        self._medivac_to_thor[medivac_tag][
                            "target"
                        ] = self.attack_target
                    else:
                        close_structures: Units = self.ai.mediator.get_units_in_range(
                            start_points=[thor.position],
                            distances=10.0,
                            query_tree=UnitTreeQueryType.AllEnemy,
                        )[0].filter(lambda u: u.type_id in ALL_STRUCTURES)
                        if not close_structures:
                            self.ai.mediator.assign_role(
                                tag=thor.tag, role=UnitRole.DROP_UNITS_TO_LOAD
                            )
                            self._medivac_to_thor[medivac_tag][
                                "target"
                            ] = self.attack_target

            if medivac and not medivac.has_cargo and not thor:
                keys_to_remove.append(medivac_tag)
                self.ai.mediator.assign_role(tag=medivac.tag, role=UnitRole.ATTACKING)

            if not medivac and thor:
                keys_to_remove.append(medivac_tag)

        for key in keys_to_remove:
            self._medivac_to_thor.pop(key)

    async def _handle_repair_crew(self):
        if not self._attack_started:
            return
        repair_crew: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.OFFENSIVE_REPAIR
        )
        if self._assign_repairers and len(repair_crew) < 5:
            if worker := self.ai.mediator.select_worker(
                target_position=self.ai.main_base_ramp.top_center
            ):
                self.ai.mediator.assign_role(
                    tag=worker.tag, role=UnitRole.OFFENSIVE_REPAIR
                )
                await self.ai.client.toggle_autocast(
                    [worker], AbilityId.EFFECT_REPAIR_SCV
                )

        if repair_crew:
            grid: np.ndarray = self.ai.mediator.get_ground_grid
            for unit in repair_crew:
                if unit.is_repairing:
                    continue
                maneuver: CombatManeuver = CombatManeuver()
                maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                maneuver.add(AMove(unit, self.healing_spot))
                unit.move(self.healing_spot)
