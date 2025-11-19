from map_analyzer import MapData, Region
from sc2.ids.ability_id import AbilityId

from ares import AresBot
from ares.behaviors.macro import BuildStructure, SpawnController
from cython_extensions import cy_distance_to_squared, cy_unit_pending
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import UnitRole

from bot.combat.base_combat import BaseCombat
from bot.combat.battle_cruiser_combat import BattleCruiserCombat
from bot.combat.ground_range_combat import GroundRangeCombat
from bot.openings.bio import Bio
from bot.openings.opening_base import OpeningBase

DEFEND_TYPES: set[UnitTypeId] = {UnitTypeId.MARINE, UnitTypeId.SIEGETANK}


class Turtle(OpeningBase):
    _bio: OpeningBase
    _ground_range_combat: BaseCombat
    _natural_hg_tank_pos: Point2

    def __init__(self):
        super().__init__()
        self._attack_started: bool = False
        self._first_tank: bool = False

    @property
    def army_comp(self) -> dict:
        if self.ai.supply_army < 30 and len(self.ai.gas_buildings) < 4:
            return {
                UnitTypeId.MARINE: {"proportion": 0.6, "priority": 1},
                UnitTypeId.SIEGETANK: {"proportion": 0.4, "priority": 0},
            }

        return {
            UnitTypeId.BATTLECRUISER: {"proportion": 0.7, "priority": 0},
            UnitTypeId.SIEGETANK: {"proportion": 0.3, "priority": 1},
        }

    @property
    def upgrade_list(self) -> list[UpgradeId]:
        required_upgrades: list[UpgradeId] = [
            UpgradeId.HISECAUTOTRACKING,
            UpgradeId.TERRANBUILDINGARMOR,
        ]
        if len(self.ai.gas_buildings) >= 4:
            required_upgrades.append(UpgradeId.BATTLECRUISERENABLESPECIALIZATIONS)
        return required_upgrades

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._battle_cruiser_combat = BattleCruiserCombat(ai, ai.config, ai.mediator)
        self._bio = Bio()
        await self._bio.on_start(ai)

        map_data: MapData = self.ai.mediator.get_map_data_object
        location = self.ai.start_location
        region: Region = map_data.in_region_p(location)
        _natural_hg_tank_pos = map_data.closest_towards_point(
            points=region.corner_points, target=self.ai.mediator.get_own_nat
        )

    async def on_step(self) -> None:
        if not self.ai.build_order_runner.build_completed:
            self.ai.register_behavior(
                SpawnController(
                    {
                        UnitTypeId.MARINE: {"proportion": 1.0, "priority": 0},
                    }
                )
            )
            return

        self._macro()
        self._add_bunkers()
        self._add_turrets()

        if self.ai.actual_iteration % 16 == 0:
            self._assign_tanks_to_bases()

    def _macro(self):
        self._generic_macro_plan(
            self.army_comp,
            self.ai.start_location,
            self.upgrade_list,
            add_hellions=False,
            add_upgrades=True,
            can_expand=len(self.ai.townhalls) < 3,
            freeflow_mode=True,
            upgrade_to_pfs=True,
            num_one_base_workers=24,
            num_gas_buildings=100 if self.ai.supply_workers > 30 else 2,
        )

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id in DEFEND_TYPES:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.DEFENDING)
            # let the siege tank just siege where it is
            if not self._first_tank and unit.type_id == UnitTypeId.SIEGETANK:
                self._first_tank = True
                unit(AbilityId.SIEGEMODE_SIEGEMODE)

    def _add_bunkers(self) -> None:
        # build one bunker at a time for simplicity
        if self.ai.structure_pending(UnitTypeId.BUNKER):
            return

        for th in self.ai.townhalls:
            location: Point2 = th.position
            existing_bunkers: list[Unit] = [
                s
                for s in self.ai.structures
                if s.type_id == UnitTypeId.BUNKER
                and cy_distance_to_squared(location, s.position) < 295.0
            ]
            if len(existing_bunkers) < 2:
                self.ai.register_behavior(BuildStructure(location, UnitTypeId.BUNKER))

    def _add_turrets(self):
        # Build missile turrets at each base
        if self.ai.minerals > 300.0 and not self.ai.structure_pending(
            UnitTypeId.MISSILETURRET
        ):
            eng_bay_ready: bool = (
                len(
                    [
                        s
                        for s in self.ai.mediator.get_own_structures_dict[
                            UnitTypeId.ENGINEERINGBAY
                        ]
                        if s.is_ready
                    ]
                )
                > 0
            )
            if not eng_bay_ready:
                return

            for townhall in self.ai.townhalls:
                if not self.ai.can_afford(
                    UnitTypeId.MISSILETURRET
                ) or self.ai.structure_pending(UnitTypeId.MISSILETURRET):
                    break

                location: Point2 = townhall.position
                existing_turrets: list[Unit] = [
                    s
                    for s in self.ai.structures
                    if s.type_id == UnitTypeId.MISSILETURRET
                    and cy_distance_to_squared(location, s.position) < 200.0
                ]
                if len(existing_turrets) < 7:
                    self.ai.register_behavior(
                        BuildStructure(
                            location, UnitTypeId.MISSILETURRET, closest_to=location
                        )
                    )

    def _assign_tanks_to_bases(self):
        pass
