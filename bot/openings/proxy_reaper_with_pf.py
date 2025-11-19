from cython_extensions import cy_distance_to_squared, cy_towards, cy_closest_to
from sc2.ids.ability_id import AbilityId
from sc2.units import Units

from ares import AresBot
from ares.behaviors.combat.individual import PickUpCargo
from ares.behaviors.macro import BuildStructure
from ares.consts import (
    LOSS_MARGINAL_OR_WORSE,
    VICTORY_CLOSE_OR_BETTER,
    EngagementResult,
    UnitRole,
)
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit

from bot.combat.base_combat import BaseCombat
from bot.openings.opening_base import OpeningBase
from bot.openings.reapers import Reapers
from bot.combat.cyclone_combat import CycloneCombat

PATH_THRESHOLD: int = 100

STATIC_DEFENCE: set[UnitTypeId] = {
    UnitTypeId.BUNKER,
    UnitTypeId.PLANETARYFORTRESS,
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.PHOTONCANNON,
}


class ProxyReaperWithPf(OpeningBase):
    _proxy_complete: bool
    _proxy_cc_complete: bool
    _proxy_location: Point2
    _reapers: OpeningBase
    _proxy_cc_location: Point2
    _cyclone_combat: BaseCombat

    SQUAD_ENGAGE_THRESHOLD: set[EngagementResult] = VICTORY_CLOSE_OR_BETTER
    SQUAD_DISENGAGE_THRESHOLD: set[EngagementResult] = LOSS_MARGINAL_OR_WORSE

    def __init__(self):
        super().__init__()
        self._pf_builder_tag: int = 0

    @property
    def army_comp(self) -> dict:
        if (
            self.ai.supply_army > 11
            or self.ai.time > 260.0
            or (self.ai.minerals > 250 and self.ai.vespene > 225)
        ):
            return {
                UnitTypeId.REAPER: {"proportion": 0.4, "priority": 1},
                UnitTypeId.CYCLONE: {"proportion": 0.6, "priority": 0},
            }
        return {
            UnitTypeId.REAPER: {"proportion": 1.0, "priority": 0},
        }

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._proxy_complete = False
        self._proxy_cc_complete = False
        self._proxy_location = self._calculate_proxy_location()
        self._proxy_cc_location = self._calculate_proxy_cc_location()

        self._reapers = Reapers()
        await self._reapers.on_start(ai)

        self._cyclone_combat = CycloneCombat(ai, ai.config, ai.mediator)

    async def on_step(self) -> None:
        await self._reapers.on_step()

        # Handle proxy construction
        proxy_scvs_amount: int = 2 if not self._proxy_complete else 0
        proxy_scvs = self._handle_proxy_scv_assignment(
            proxy_scvs_amount, self._proxy_location
        )
        if proxy_scvs:
            await self.proxy_construction_manager.handle_construction(
                proxy_scvs, self._proxy_location, UnitTypeId.BARRACKS, 2
            )

        if (
            len(
                [
                    b
                    for b in self.ai.mediator.get_own_structures_dict[
                        UnitTypeId.BARRACKS
                    ]
                    if b.is_ready
                ]
            )
            >= 2
        ):
            self._proxy_complete = True

        if self.ai.build_order_runner.build_completed and (
            len(self.ai.mediator.get_own_structures_dict[UnitTypeId.BARRACKS]) >= 1
            or self.ai.time > 180.0
        ):
            self._macro()

        if self._proxy_complete and not self._proxy_cc_complete:
            self._handle_proxy_cc_construction()

        await self._control_flying_cc()

        self._cyclone_combat.execute(
            self.ai.mediator.get_units_from_role(role=UnitRole.ATTACKING),
            target=self.attack_target,
        )

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.CYCLONE:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    def _macro(self):
        own_rax = self.ai.mediator.get_own_structures_dict[UnitTypeId.BARRACKS]
        self._generic_macro_plan(
            self.army_comp,
            self.ai.start_location,
            [],
            add_hellions=False,
            add_upgrades=True,
            can_expand=self.ai.supply_army > 15 or self.ai.minerals > 410,
            freeflow_mode=True,
            upgrade_to_pfs=False,
            max_pending_gas_buildings=1,
            num_gas_buildings=2 if self.ai.supply_army < 16 else 100,
            num_one_base_workers=14 if len(own_rax) < 2 else 18,
            max_workers=66,
            can_add_orbital=len([r for r in own_rax if not r.is_active]) == 0
            and len(self.ai.townhalls) < 2,
            production_controller_enabled=True,
        )

        if (
            not self.ai.structure_present_or_pending(UnitTypeId.ENGINEERINGBAY)
            and len(self.ai.townhalls) > 1
        ):
            self.ai.register_behavior(
                BuildStructure(self.ai.start_location, UnitTypeId.ENGINEERINGBAY)
            )

    def _handle_proxy_cc_construction(self) -> None:
        scv: Unit | None = self.ai.unit_tag_dict.get(self._pf_builder_tag)
        if not scv:
            if _scv := self.ai.mediator.select_worker(
                target_position=self._proxy_location
            ):
                self._pf_builder_tag = _scv.tag
                self.ai.mediator.assign_role(
                    tag=_scv.tag, role=UnitRole.CONTROL_GROUP_NINE
                )
                scv = _scv

        if not scv or scv.is_constructing_scv:
            return

        # check for completion
        if len(self.ai.townhalls.ready) >= 2:
            self._proxy_cc_complete = True
            self._pf_builder_tag = 0
            self.ai.mediator.assign_role(tag=scv.tag, role=UnitRole.GATHERING)
            return

        if self.ai.can_afford(UnitTypeId.COMMANDCENTER):
            scv.build(UnitTypeId.COMMANDCENTER, self._proxy_cc_location)
        else:
            scv.move(self._proxy_cc_location)

    def _calculate_proxy_cc_location(self) -> Point2:
        potential_locations: list[
            tuple[Point2, float]
        ] = self.ai.mediator.get_enemy_expansions[1:6]

        closest = potential_locations[0]
        closest_dist: float = 998000.0
        target: Point2 = self.ai.enemy_start_locations[0]

        for loc in potential_locations:
            dist: float = cy_distance_to_squared(loc[0], target)
            if dist < closest_dist:
                closest_dist = dist
                closest = loc[0]

        return closest

    async def _control_flying_cc(self):
        if self.ai.time < 150.0:
            return

        ccs: list[Unit] = self.ai.mediator.get_own_structures_dict[
            UnitTypeId.COMMANDCENTER
        ]
        target: Point2 = Point2(
            cy_towards(self.ai.enemy_start_locations[0], self._proxy_cc_location, 6.0)
        )
        ready_ccs: list[Unit] = [
            s
            for s in ccs
            if s.is_ready
            and cy_distance_to_squared(s.position, target) > 380.0
            and cy_distance_to_squared(s.position, self.ai.start_location) > 200.0
        ]

        if len(ready_ccs) > 0:
            cc: Unit = ready_ccs[0]
            cc(AbilityId.LIFT_COMMANDCENTER)

        flying_ccs: list[Unit] = self.ai.mediator.get_own_structures_dict[
            UnitTypeId.COMMANDCENTERFLYING
        ]

        for flying_cc in flying_ccs:
            if flying_cc.is_using_ability(AbilityId.LAND):
                continue
            if cy_distance_to_squared(flying_cc.position, target) < 300.0:
                placement: Point2 | None = await self.ai.find_placement(
                    UnitTypeId.COMMANDCENTER, target, placement_step=1
                )
                if placement:
                    flying_cc(AbilityId.LAND_COMMANDCENTER, placement)
                    break

            flying_cc.move(target)

        near_to_location: list[Unit] = [
            s
            for s in ccs
            if s.is_ready and cy_distance_to_squared(s.position, target) < 300
        ]
        for n_cc in near_to_location:
            if self.ai.can_afford(UnitTypeId.PLANETARYFORTRESS):
                n_cc(AbilityId.UPGRADETOPLANETARYFORTRESS_PLANETARYFORTRESS)
