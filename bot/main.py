import importlib
import math
from typing import Any, Optional

import numpy as np
from loguru import logger
from sc2.position import Point2

from ares import AresBot
from ares.behaviors.combat.individual import KeepUnitSafe
from ares.behaviors.macro.mining import Mining
from ares.consts import ALL_STRUCTURES, UnitRole
from cython_extensions import cy_distance_to_squared, cy_closest_to, cy_towards
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.unit import Unit

from bot.consts import UNIT_TYPE_TO_NUM_REPAIRERS


def _to_snake(name: str) -> str:
    # Convert e.g. "OneBaseTempest" -> "one_base_tempest"
    out = []
    for i, c in enumerate(name):
        if i > 0:
            prev = name[i - 1]
            nxt = name[i + 1] if i + 1 < len(name) else ""
            if c.isupper() and (
                (not prev.isupper())  # lower->Upper
                or (prev.isupper() and nxt and not nxt.isupper())  # UPPER->UpperLower
            ):
                out.append("_")
        out.append(c.lower())
    return "".join(out)


class MyBot(AresBot):
    def __init__(self, game_step_override: Optional[int] = None):
        """Initiate custom bot

        Parameters
        ----------
        game_step_override :
            If provided, set the game_step to this value regardless of how it was
            specified elsewhere
        """
        super().__init__(game_step_override)
        self.opening_handler: Optional[Any] = None
        self.opening_chat_tag: bool = False
        self._switched_to_prevent_tie: bool = False
        self.injured_general_unit_to_repairing_scvs: dict[int, set[int]] = dict()
        self._terran_bunker_finder_activated: bool = False

    def load_opening(self, opening_name: str) -> None:
        """Load opening from bot.openings.<snake_case> with class <PascalCase>"""
        module_path = f"bot.openings.{_to_snake(opening_name)}"
        module = importlib.import_module(module_path)
        opening_cls = getattr(module, opening_name, None)
        if opening_cls is None:
            raise ImportError(
                f"Opening class '{opening_name}' not found in '{module_path}'"
            )
        self.opening_handler = opening_cls()

    async def on_start(self) -> None:
        await super(MyBot, self).on_start()
        # Ares has initialized BuildOrderRunner at this point
        try:
            self.load_opening(self.build_order_runner.chosen_opening)
            if hasattr(self.opening_handler, "on_start"):
                await self.opening_handler.on_start(self)
        except Exception as exc:
            print(f"Failed to load opening: {exc}")

    async def on_step(self, iteration: int) -> None:
        await super(MyBot, self).on_step(iteration)
        if self.supply_used < 1:
            await self.client.leave()
        self.register_behavior(Mining())

        self._mules()
        self._general_repair()
        self._look_for_terran_bunker()

        if self.opening_handler and hasattr(self.opening_handler, "on_step"):
            await self.opening_handler.on_step()

        if not self.opening_chat_tag and self.time > 5.0:
            await self.chat_send(
                f"Tag: {self.build_order_runner.chosen_opening}", team_only=True
            )
            self.opening_chat_tag = True

        if not self._switched_to_prevent_tie and self.floating_enemy:
            self._switched_to_prevent_tie = True
            self.load_opening("BattleCruiserRush")
            if hasattr(self.opening_handler, "on_start"):
                await self.opening_handler.on_start(self)
            for worker in self.workers:
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)

            await self.chat_send(f"Tag: {self.time_formatted}_switched_to_prevent_tie")

        if iteration % 16 == 0:
            own_structures_dict = self.mediator.get_own_structures_dict
            depots: list[Unit] = own_structures_dict[UnitTypeId.SUPPLYDEPOT]
            for depot in depots:
                if not depot.is_ready:
                    continue
                if depot.type_id == UnitTypeId.SUPPLYDEPOT:
                    depot(AbilityId.MORPH_SUPPLYDEPOT_LOWER)

    async def on_unit_created(self, unit: Unit) -> None:
        await super(MyBot, self).on_unit_created(unit)
        if unit.type_id == UnitTypeId.REAPER:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.HARASSING_REAPER)

        if self.opening_handler and hasattr(self.opening_handler, "on_unit_created"):
            self.opening_handler.on_unit_created(unit)

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)

        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            self.mediator.cancel_structure(structure=unit)

    def _general_repair(self) -> None:
        self._execute_scv_to_general_repair()

        for unit in self.all_own_units:
            type_id: UnitTypeId = unit.type_id
            if (
                unit.health_percentage >= 1.0
                or not unit.is_ready
                or (
                    type_id not in ALL_STRUCTURES
                    and cy_distance_to_squared(unit.position, self.start_location)
                    > 1600
                )
                or type_id not in UNIT_TYPE_TO_NUM_REPAIRERS
            ):
                continue
            if type_id in ALL_STRUCTURES and unit.health_percentage > 0.95:
                continue

            if type_id in UNIT_TYPE_TO_NUM_REPAIRERS:
                if type_id == UnitTypeId.HELLION and self.enemy_race == Race.Terran:
                    continue
                num_scvs_required: int = UNIT_TYPE_TO_NUM_REPAIRERS[unit.type_id]
                if unit.tag in self.injured_general_unit_to_repairing_scvs:
                    num_scvs_required -= len(
                        self.injured_general_unit_to_repairing_scvs[unit.tag]
                    )
                for _ in range(num_scvs_required):
                    if worker := self.mediator.select_worker(
                        target_position=unit.position,
                        force_close=True,
                        min_health_perc=0.45,
                    ):
                        if unit.tag in self.injured_general_unit_to_repairing_scvs:
                            self.injured_general_unit_to_repairing_scvs[unit.tag].add(
                                worker.tag
                            )
                        else:
                            self.injured_general_unit_to_repairing_scvs[unit.tag] = {
                                worker.tag
                            }
                        self.mediator.assign_role(
                            tag=worker.tag, role=UnitRole.REPAIRING
                        )

    def _execute_scv_to_general_repair(self):
        """ """
        remove_tags: list[int] = []
        remove_medics: dict[int, int] = dict()
        for (
            injured_tag,
            medic_tags,
        ) in self.injured_general_unit_to_repairing_scvs.items():
            injured: Unit = self.unit_tag_dict.get(injured_tag, None)
            # injured / medic ded? low income? :(
            # or both full health? :D
            if not injured or injured.health_percentage >= 1.0:
                remove_tags.append(injured_tag)
                continue

            medics: list[Unit] = []
            for tag in medic_tags:
                medic: Optional[Unit] = self.unit_tag_dict.get(tag, None)
                if (
                    not medic
                    or medic.health_percentage < 0.4
                    and injured_tag not in remove_tags
                ):
                    remove_medics[injured_tag] = tag
                else:
                    medics.append(medic)

            # got here, we are alive and well :D
            # do repair logic
            self._scvs_to_general_repair_logic(injured, medics)

        for tag in remove_tags:
            medic_tags: set[int] = self.injured_general_unit_to_repairing_scvs[tag]
            self.mediator.batch_assign_role(tags=medic_tags, role=UnitRole.GATHERING)
            self.mediator.assign_role(tag=tag, role=UnitRole.ATTACKING)
            self.injured_general_unit_to_repairing_scvs.pop(tag)

        for tag, remove_tag in remove_medics.items():
            self.injured_general_unit_to_repairing_scvs[tag].remove(remove_tag)

    def _scvs_to_general_repair_logic(self, injured: Unit, medics: list[Unit]) -> None:
        grid: np.ndarray = self.mediator.get_ground_avoidance_grid

        for medic in medics:
            # avoid biles etc
            if not self.mediator.is_position_safe(grid=grid, position=medic.position):
                self.register_behavior(KeepUnitSafe(medic, grid))
                continue

            if medic.is_repairing:
                continue

            medic(AbilityId.EFFECT_REPAIR_SCV, injured)

    def _mules(self):
        oc_id: UnitTypeId = UnitTypeId.ORBITALCOMMAND
        structures_dict: dict[
            UnitTypeId, list[Unit]
        ] = self.mediator.get_own_structures_dict
        for oc in [s for s in structures_dict[oc_id] if s.energy >= 50]:
            mfs: list[Unit] = [
                mf
                for mf in self.mineral_field
                if cy_distance_to_squared(mf.position, oc.position) < 100.0
            ]
            if mfs:
                mf: Unit = max(mfs, key=lambda x: x.mineral_contents)
                oc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, mf)

    def _look_for_terran_bunker(self):
        # when core is ready have a look around for proxies
        if (
            self.enemy_race == Race.Terran
            and not self._terran_bunker_finder_activated
            and self.time > 77.0
        ):
            natural_location: Point2 = self.mediator.get_own_nat
            if worker := self.mediator.select_worker(target_position=natural_location):
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.MAP_CONTROL)

                radius = 9
                num_points = 10

                # Generate the scouting positions in a circular pattern
                scouting_positions = [
                    Point2(
                        (
                            natural_location.x
                            + radius * math.cos(2 * math.pi * i / num_points),
                            natural_location.y
                            + radius * math.sin(2 * math.pi * i / num_points),
                        )
                    )
                    for i in range(num_points)
                ]
                scouting_positions = [
                    s for s in scouting_positions if self.in_pathing_grid(s)
                ]

                for i, position in enumerate(scouting_positions):
                    worker.move(position, queue=i != 0)

                logger.info(
                    f"{self.time_formatted} - Sent scout to circle around natural"
                )

                self._terran_bunker_finder_activated = True
                return

        if self._terran_bunker_finder_activated and (
            scouts := self.mediator.get_units_from_role(
                role=UnitRole.MAP_CONTROL, unit_type=UnitTypeId.SCV
            )
        ):
            for scout in scouts:
                if proxies := self.get_enemy_proxies(30.0, scout.position):
                    scout.attack(cy_closest_to(scout.position, proxies).position)

                elif self.mediator.get_enemy_expanded:
                    self.mediator.assign_role(tag=scout.tag, role=UnitRole.GATHERING)

                elif scout.is_idle:
                    if self.time < 125.0:
                        scout.move(
                            Point2(
                                cy_towards(
                                    self.mediator.get_own_nat,
                                    self.game_info.map_center,
                                    10.0,
                                )
                            )
                        )
                    else:
                        self.mediator.assign_role(
                            tag=scout.tag, role=UnitRole.GATHERING
                        )

    @property
    def floating_enemy(self) -> bool:
        if self.enemy_race != Race.Terran or self.time < 270.0:
            return False

        if (
            len([s for s in self.enemy_structures if s.is_flying]) > 0
            and self.state.visibility[self.enemy_start_locations[0].rounded] != 0
            and len(self.enemy_units) < 4
        ):
            return True

        return False

    """
    Can use `python-sc2` hooks as usual, but make a call the inherited method in the superclass
    Examples:
    """

    #
    # async def on_end(self, game_result: Result) -> None:
    #     await super(MyBot, self).on_end(game_result)
    #
    #     # custom on_end logic here ...
    #
    # async def on_building_construction_complete(self, unit: Unit) -> None:
    #     await super(MyBot, self).on_building_construction_complete(unit)
    #
    #     # custom on_building_construction_complete logic here ...
    #
    # async def on_unit_created(self, unit: Unit) -> None:
    #     await super(MyBot, self).on_unit_created(unit)
    #
    #     # custom on_unit_created logic here ...
    #
    # async def on_unit_destroyed(self, unit_tag: int) -> None:
    #     await super(MyBot, self).on_unit_destroyed(unit_tag)
    #
    #     # custom on_unit_destroyed logic here ...
    #
    # async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
    #     await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)
    #
    #     # custom on_unit_took_damage logic here ...
