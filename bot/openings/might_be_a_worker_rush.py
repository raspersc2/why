from sc2.ids.ability_id import AbilityId

from ares import AresBot
from cython_extensions import (
    cy_closest_to,
    cy_distance_to_squared,
    cy_center,
)
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares.behaviors.combat.individual import PathUnitToTarget
from ares.cache import property_cache_once_per_frame
from ares.managers.squad_manager import UnitSquad
from bot.consts import COMMON_UNIT_IGNORE_TYPES
from src.ares.consts import UnitRole, UnitTreeQueryType

from bot.combat.base_combat import BaseCombat
from bot.combat.worker_combat import WorkerCombat
from bot.openings.bio import Bio
from bot.openings.opening_base import OpeningBase
from bot.openings.worker_rush import WorkerRush


class MightBeAWorkerRush(OpeningBase):
    _bio: OpeningBase
    _worker_rush: OpeningBase

    def __init__(self):
        super().__init__()

    @property
    def army_comp(self) -> dict:
        return {
            UnitTypeId.MARINE: {"proportion": 1.0, "priority": 0},
        }

    @property_cache_once_per_frame
    def healing_spot(self) -> Point2:
        return self.ai.mediator.find_closest_safe_spot(
            from_pos=self.target_healing_pos,
            grid=self.ai.mediator.get_ground_grid,
            radius=15.0,
        )

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._bio = Bio()
        await self._bio.on_start(ai)
        self._worker_rush = WorkerRush()
        await self._worker_rush.on_start(ai)

    async def on_step(self) -> None:
        if (
            self.ai.build_order_runner.build_completed
            and len(self.ai.mediator.get_own_structures_dict[UnitTypeId.BARRACKS]) > 0
        ):
            self._macro()

        await self._worker_rush.on_step()
        attack_target: Point2 = self.attack_target
        await self._bio.on_step(target=attack_target)

        await self._handle_proxy_rax()

    def _macro(self):
        self._generic_macro_plan(
            self.army_comp,
            self.ai.start_location,
            [],
            add_hellions=False,
            add_upgrades=False,
            can_expand=False,
            freeflow_mode=True,
            upgrade_to_pfs=False,
            num_one_base_workers=14,
            num_gas_buildings=0,
            can_add_orbital=False,
        )

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.MARINE:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    async def _handle_proxy_rax(self):
        workers: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.PROXY_WORKER
        )
        if not workers:
            if unfinished_rax := self.ai.structures(UnitTypeId.BARRACKS).not_ready:
                self.ai.mediator.cancel_structure(structure=unfinished_rax[0])
            return

        worker: Unit = workers[0]
        if worker.is_carrying_resource:
            worker.return_resource()
        elif worker.is_constructing_scv:
            return
        elif (
            cy_distance_to_squared(
                worker.position, self.ai.mediator.get_primary_nydus_enemy_main
            )
            < 25.0
        ):
            if not worker.orders and self.ai.can_afford(UnitTypeId.BARRACKS):
                if pos := await self.ai.find_placement(
                    UnitTypeId.BARRACKS,
                    self.ai.mediator.get_primary_nydus_enemy_main,
                    placement_step=1,
                ):
                    worker.build(UnitTypeId.BARRACKS, pos)
        else:
            self.ai.register_behavior(
                PathUnitToTarget(
                    unit=worker,
                    target=self.ai.mediator.get_primary_nydus_enemy_main,
                    grid=self.ai.mediator.get_ground_grid,
                )
            )

    def on_building_construction_complete(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.BARRACKS:
            self.ai.mediator.switch_roles(
                from_role=UnitRole.PROXY_WORKER, to_role=UnitRole.CONTROL_GROUP_EIGHT
            )
