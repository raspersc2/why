import numpy as np
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

from ares.behaviors.combat.individual import WorkerKiteBack, KeepUnitSafe
from ares.managers.squad_manager import UnitSquad
from bot.consts import COMMON_UNIT_IGNORE_TYPES
from src.ares.consts import UnitRole, UnitTreeQueryType

from bot.combat.base_combat import BaseCombat
from bot.combat.worker_combat import WorkerCombat
from bot.openings.bio import Bio
from bot.openings.opening_base import OpeningBase


class WorkerRush(OpeningBase):
    _bio: OpeningBase
    _worker_combat: BaseCombat
    target_healing_pos: Point2

    def __init__(self):
        super().__init__()
        self._attack_started: bool = False
        self._start_attack_at_time: float = 13.4
        self._initial_assignment: bool = False
        self._max_scvs_in_attack: int = 9
        self._stack_for: float = 1.85
        self._low_health_tags: set[int] = set()
        self._enemy_walled_off: bool = False

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._bio = Bio()
        await self._bio.on_start(ai)

        self._worker_combat = WorkerCombat(ai, ai.config, ai.mediator)
        self.target_healing_pos = self.ai.game_info.map_center

        if self.ai.build_order_runner.chosen_opening == "WorkerRush":
            self.ai.client.game_step = 1
            self._start_attack_at_time = 7
            self._max_scvs_in_attack = 11
        elif self.ai.build_order_runner.chosen_opening != "MightBeAWorkerRush":
            self._max_scvs_in_attack = 15

    async def on_step(self) -> None:
        if not self._attack_started:
            if self.ai.time >= self._start_attack_at_time:
                self._attack_started = True
            return

        if not self._enemy_walled_off and self.ai.state.game_loop % 20 == 0:
            self._enemy_walled_off = self.ai.main_ramp_walled_off(
                self.ai.mediator.get_enemy_ramp
            )
            if self._enemy_walled_off:
                await self.ai.chat_send(f"Tag: {self.ai.time_formatted}: WallOff")

        await self._assign_workers()

        self._handle_worker_repair()
        squads: list[UnitSquad] = self.ai.mediator.get_squads(
            role=UnitRole.CONTROL_GROUP_EIGHT, squad_radius=9.0
        )
        if len(squads) == 0:
            return

        pos_of_main_squad: Point2 = self.ai.mediator.get_position_of_main_squad(
            role=UnitRole.CONTROL_GROUP_EIGHT
        )
        for squad in squads:
            if self.ai.time < self._start_attack_at_time + self._stack_for:
                mf: Unit = cy_closest_to(self.ai.start_location, self.ai.mineral_field)
                for unit in squad.squad_units:
                    if unit.is_carrying_resource:
                        unit.return_resource()
                    else:
                        unit.gather(mf)
                continue

            if self.ai.time < 150.0 and (
                flying_structures := [
                    s
                    for s in self.ai.enemy_structures
                    if s.is_flying and not s.is_memory
                ]
            ):
                target: Point2 = flying_structures[0].position
            else:
                target: Point2 = (
                    self.attack_target if squad.main_squad else pos_of_main_squad
                )
            close_ground_enemy: Units = self.ai.mediator.get_units_in_range(
                start_points=[squad.squad_position],
                distances=12.5,
                query_tree=UnitTreeQueryType.EnemyGround,
            )[0].filter(lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES)
            self._worker_combat.execute(
                units=squad.squad_units,
                all_close_enemy=close_ground_enemy,
                target=target,
                ramp_walled_off=self._enemy_walled_off,
            )

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.MARINE:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    async def _assign_workers(self):
        if not self._initial_assignment:
            num_assigned: int = 0
            for worker in self.ai.mediator.get_units_from_role(role=UnitRole.GATHERING):
                if num_assigned >= self._max_scvs_in_attack:
                    break

                role: UnitRole = (
                    UnitRole.PROXY_WORKER
                    if num_assigned == 0
                    and self.ai.build_order_runner.chosen_opening != "WorkerRush"
                    else UnitRole.CONTROL_GROUP_EIGHT
                )

                self.ai.mediator.assign_role(tag=worker.tag, role=role)
                self.ai.mediator.remove_worker_from_mineral(worker_tag=worker.tag)
                await self.ai.client.toggle_autocast(
                    [worker], AbilityId.EFFECT_REPAIR_SCV
                )
                num_assigned += 1
            self._initial_assignment = True

    def _handle_worker_repair(self):
        all_workers: Units = self.ai.mediator.get_units_from_roles(
            roles={UnitRole.CONTROL_GROUP_ONE, UnitRole.CONTROL_GROUP_EIGHT}
        )
        healing: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.CONTROL_GROUP_ONE
        )
        grid: np.ndarray = self.ai.mediator.get_ground_grid
        for worker in all_workers:
            health_perc: float = worker.health_percentage
            if health_perc < 0.4 and len(all_workers) >= 3 and self.ai.minerals > 0:
                self._low_health_tags.add(worker.tag)
                self.ai.mediator.assign_role(
                    tag=worker.tag, role=UnitRole.CONTROL_GROUP_ONE
                )
            elif worker.tag in self._low_health_tags and (
                health_perc >= 1.0
                or len(all_workers) < 3
                or (len(healing) == 1 and health_perc >= 0.6)
                or self.ai.minerals <= 0
            ):
                self._low_health_tags.remove(worker.tag)
                self.ai.mediator.assign_role(
                    tag=worker.tag, role=UnitRole.CONTROL_GROUP_EIGHT
                )

            if worker.tag in self._low_health_tags:
                nearby_units: list[Unit] = [
                    w
                    for w in self.ai.all_units
                    if cy_distance_to_squared(w.position, worker.position) < 7.5
                    and w.tag != worker.tag
                    and w.tag not in self._low_health_tags
                ]
                if len(nearby_units) >= 4:
                    self.ai.register_behavior(
                        WorkerKiteBack(worker, nearby_units[0], should_attack=False)
                    )

                else:
                    if KeepUnitSafe(worker, grid).execute(
                        self.ai, self.ai.config, self.ai.mediator
                    ):
                        continue
                    close_healing: list[Unit] = [
                        u
                        for u in healing
                        if cy_distance_to_squared(worker.position, u.position) < 25.0
                    ]
                    if len(healing) == 1:
                        KeepUnitSafe(worker, grid).execute(
                            self.ai, self.ai.config, self.ai.mediator
                        )
                    elif len(close_healing) >= 2:
                        worker(AbilityId.STOP)
                    else:
                        worker.move(Point2(cy_center(healing)))
