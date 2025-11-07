from abc import ABCMeta, abstractmethod
from itertools import cycle

import numpy as np

from ares import AresBot
from ares.behaviors.macro import (
    AutoSupply,
    BuildStructure,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    ProductionController,
    SpawnController,
    UpgradeCCs,
    UpgradeController,
)
from ares.cache import property_cache_once_per_frame
from ares.consts import UnitRole
from cython_extensions import cy_find_units_center_mass
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.units import Units

from bot.consts import ATTACK_TARGET_IGNORE


class OpeningBase(metaclass=ABCMeta):
    ai: AresBot

    def __init__(self):
        super().__init__()
        self.expansions_generator = None
        self.current_base_target: Point2 = Point2((0, 0))

    @abstractmethod
    async def on_start(self, ai: AresBot) -> None:
        self.ai = ai
        self.current_base_target = ai.enemy_start_locations[0]

    @abstractmethod
    async def on_step(self, target: Point2 | None = None) -> None:
        pass

    def _calculate_proxy_location(self) -> Point2:
        potential_locations: list[
            tuple[Point2, float]
        ] = self.ai.mediator.get_enemy_expansions[2:6]

        closest = potential_locations[0]
        closest_dist: float = 998000.0
        grid: np.ndarray = self.ai.mediator.get_ground_grid
        target: Point2 = self.ai.enemy_start_locations[0]

        for loc in potential_locations:
            if path := self.ai.mediator.find_raw_path(
                start=target, target=loc[0], grid=grid, sensitivity=2
            ):
                dist: int = len(path)
                if dist < closest_dist:
                    closest_dist = dist
                    closest = loc[0]

        return closest

    def _generic_macro_plan(
        self,
        army_comp: dict,
        build_location: Point2,
        upgrades: list[UpgradeId],
        add_hellions: bool = True,
        add_upgrades: bool = True,
        freeflow_mode: bool = True,
        can_expand: bool = True,
        upgrade_to_pfs: bool = True,
        production_controller_enabled: bool = True,
        num_one_base_workers: int = 20,
        num_gas_buildings: int = 100,
    ) -> None:
        if (
            upgrade_to_pfs
            and not self.ai.structure_present_or_pending(UnitTypeId.ENGINEERINGBAY)
            and len(self.ai.townhalls) > 1
        ):
            self.ai.register_behavior(
                BuildStructure(build_location, UnitTypeId.ENGINEERINGBAY)
            )

        macro_plan: MacroPlan = MacroPlan()

        if production_controller_enabled:
            macro_plan.add(ProductionController(army_comp, build_location))

        macro_plan.add(AutoSupply(self.ai.start_location))
        macro_plan.add(GasBuildingController(num_gas_buildings))
        if (
            upgrade_to_pfs
            and self.ai.mediator.get_own_structures_dict[UnitTypeId.ENGINEERINGBAY]
        ):
            macro_plan.add(UpgradeCCs(UnitTypeId.PLANETARYFORTRESS, prioritize=True))
        else:
            macro_plan.add(UpgradeCCs(UnitTypeId.ORBITALCOMMAND, prioritize=True))

        if add_upgrades:
            macro_plan.add(UpgradeController(upgrades, build_location))

        macro_plan.add(
            SpawnController(
                army_composition_dict=army_comp, freeflow_mode=freeflow_mode
            )
        )

        if add_hellions:
            macro_plan.add(
                SpawnController(
                    army_composition_dict={
                        UnitTypeId.HELLION: {"proportion": 1.0, "priority": 0}
                    },
                    freeflow_mode=freeflow_mode,
                )
            )

        num_workers: int = (
            num_one_base_workers
            if len(self.ai.townhalls) <= 1
            else (min(60, len(self.ai.townhalls) * 22))
        )

        macro_plan.add(BuildWorkers(num_workers))

        if can_expand:
            macro_plan.add(ExpansionController(100))
        self.ai.register_behavior(macro_plan)

    def _handle_proxy_scv_assignment(
        self, max_proxy_workers: int, proxy_location: Point2
    ) -> Units:
        proxy_workers: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.PROXY_WORKER
        )
        if len(proxy_workers) > max_proxy_workers:
            for worker in proxy_workers:
                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)

        if len(proxy_workers) < max_proxy_workers:
            if worker := self.ai.mediator.select_worker(target_position=proxy_location):
                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.PROXY_WORKER)

        return proxy_workers

    @property_cache_once_per_frame
    def attack_target(self) -> Point2:
        enemy_units: Units = self.ai.enemy_units.filter(
            lambda u: u.type_id not in ATTACK_TARGET_IGNORE
            and not u.is_flying
            and not u.is_cloaked
            and not u.is_hallucination
        )
        num_units: int = 0
        center_mass: Point2 = self.ai.start_location
        if enemy_units:
            center_mass, num_units = cy_find_units_center_mass(enemy_units, 12.5)
        enemy_structures: Units = self.ai.enemy_structures
        if num_units > 5:
            return Point2(center_mass)
        elif enemy_structures and self.ai.time > 120.0:
            return enemy_structures.closest_to(self.ai.start_location).position
        elif (
            self.ai.time < 150.0
            or self.ai.state.visibility[self.ai.enemy_start_locations[0].rounded] == 0
        ):
            return self.ai.enemy_start_locations[0]
        else:
            # cycle through base locations
            if self.ai.is_visible(self.current_base_target):
                if not self.expansions_generator:
                    base_locations: list[Point2] = [
                        i for i in self.ai.expansion_locations_list
                    ]
                    self.expansions_generator = cycle(base_locations)

                self.current_base_target = next(self.expansions_generator)

            return self.current_base_target
