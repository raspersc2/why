from abc import ABCMeta, abstractmethod
from itertools import cycle

import numpy as np
from ares.consts import UnitRole

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
from cython_extensions import cy_find_units_center_mass
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.units import Units

from bot.openings.proxy_construction_manager import ProxyConstructionManager
from bot.consts import ATTACK_TARGET_IGNORE


class OpeningBase(metaclass=ABCMeta):
    ai: AresBot

    def __init__(self):
        super().__init__()
        self.expansions_generator = None
        self.current_base_target: Point2 = Point2((0, 0))
        self.proxy_construction_manager: ProxyConstructionManager | None = None

    @abstractmethod
    async def on_start(self, ai: AresBot) -> None:
        self.ai = ai
        self.current_base_target = ai.enemy_start_locations[0]
        self.proxy_construction_manager = ProxyConstructionManager(ai)

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
        max_workers: int = 60,
        num_gas_buildings: int = 100,
        max_pending_gas_buildings: int = 1,
        can_add_orbital: bool = True,
        add_production_at_bank: tuple[int, int] = (700, 700),
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
            macro_plan.add(
                ProductionController(
                    army_comp,
                    build_location,
                    add_production_at_bank=add_production_at_bank,
                )
            )

        macro_plan.add(AutoSupply(self.ai.start_location))
        macro_plan.add(
            GasBuildingController(
                num_gas_buildings, max_pending=max_pending_gas_buildings
            )
        )
        if (
            upgrade_to_pfs
            and self.ai.mediator.get_own_structures_dict[UnitTypeId.ENGINEERINGBAY]
        ):
            macro_plan.add(UpgradeCCs(UnitTypeId.PLANETARYFORTRESS, prioritize=True))
        elif can_add_orbital:
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
            else (min(max_workers, len(self.ai.townhalls) * 22))
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

    # async def _handle_proxy_construction(
    #     self,
    #     proxy_scvs: Units,
    #     proxy_location: Point2,
    #     to_build: UnitTypeId = UnitTypeId.BARRACKS,
    # ) -> None:
    #     # First pass: detect dead SCVs and find unfinished structures
    #     proxy_scv_tags: set[int] = {scv.tag for scv in proxy_scvs}
    #     unfinished_structures: list[dict] = []
    #     dead_scv_tags: list[int] = []
    #
    #     for tag, data in list(self._proxy_tracker.items()):
    #         # Check if SCV is dead (not in current proxy_scvs)
    #         if tag not in proxy_scv_tags:
    #             target_pos: Point2 = data["pos"]
    #             # Check if there's an unfinished structure at this position
    #             structures_at_pos: list[Unit] = [
    #                 s
    #                 for s in self.ai.structures
    #                 if cy_distance_to_squared(s.position, target_pos) < 9.0
    #                 and not s.is_ready
    #             ]
    #
    #             if structures_at_pos:
    #                 # Structure exists but is incomplete - needs a new builder
    #                 unfinished_structures.append({
    #                     "to_build": data["to_build"],
    #                     "pos": target_pos,
    #                     "structure": structures_at_pos[0],
    #                 })
    #                 # DEBUG!
    #                 self.ai.client.debug_text_screen(
    #                     f"Unfinished structure found at {target_pos}, needs builder",
    #                     pos=(0.1, 0.3),
    #                     size=10,
    #                 )
    #
    #             # Mark old tag for removal
    #             dead_scv_tags.append(tag)
    #
    #     # Remove dead SCV entries
    #     for tag in dead_scv_tags:
    #         # DEBUG!
    #         self.ai.client.debug_text_screen(
    #             f"Removing dead SCV {tag} from tracker",
    #             pos=(0.1, 0.35),
    #             size=10,
    #         )
    #
    #         del self._proxy_tracker[tag]
    #
    #     # Second pass: assign unfinished structures to available SCVs first
    #     for unfinished_data in unfinished_structures:
    #         # Find an available proxy SCV without a task
    #         available_scv = None
    #         for scv in proxy_scvs:
    #             if scv.tag not in self._proxy_tracker:
    #                 available_scv = scv
    #                 break
    #
    #         if available_scv:
    #             # Assign this SCV to the unfinished structure
    #             self._proxy_tracker[available_scv.tag] = {
    #                 "to_build": unfinished_data["to_build"],
    #                 "pos": unfinished_data["pos"],
    #                 "ts": self.ai.time,
    #                 "status": ProxySCVStatus.Moving,
    #             }
    #             # DEBUG!
    #             self.ai.client.debug_text_screen(
    #                 f"Assigned SCV {available_scv.tag} to unfinished structure at {unfinished_data['pos']}",
    #                 pos=(0.1, 0.4),
    #                 size=10,
    #                 )
    #         # DEBUG!
    #         else:
    #             self.ai.client.debug_text_screen(
    #                 f"No available SCV found for unfinished structure at {unfinished_data['pos']}",
    #                 pos=(0.1, 0.4),
    #                 size=10,
    #             )
    #
    #     # Third pass: handle all proxy SCVs
    #     for scv in proxy_scvs:
    #         tag: int = scv.tag
    #         if tag not in self._proxy_tracker:
    #             # This is a new SCV without a task - assign it a new build location
    #             if placement := self.ai.mediator.request_building_placement(
    #                 base_location=proxy_location,
    #                 structure_type=to_build,
    #                 closest_to=self.ai.enemy_start_locations[0],
    #             ):
    #                 self._proxy_tracker[tag] = {
    #                     "to_build": to_build,
    #                     "pos": placement,
    #                     "ts": self.ai.time,
    #                     "status": ProxySCVStatus.Moving,
    #                 }
    #                 # DEBUG!
    #                 self.ai.client.debug_text_screen(
    #                     f"Assigned NEW placement {placement} to SCV {tag}",
    #                     pos=(0.1, 0.45),
    #                     size=10,
    #                 )
    #             continue
    #
    #         # DEBUG!
    #         current_pos = self._proxy_tracker[tag]["pos"]
    #         current_status = self._proxy_tracker[tag]["status"]
    #         self.ai.client.debug_text_screen(
    #             f"SCV {tag}: status={current_status}, target={current_pos}",
    #             pos=(0.1, 0.5 + len([t for t in self._proxy_tracker.keys() if t <= tag]) * 0.03),
    #             size=8,
    #         )
    #
    #         current_status: ProxySCVStatus = self._proxy_tracker[tag]["status"]
    #         match current_status:
    #             case ProxySCVStatus.Moving:
    #                 target_pos: Point2 = self._proxy_tracker[tag]["pos"]
    #                 to_build: UnitTypeId = self._proxy_tracker[tag]["to_build"]
    #
    #                 # Check if structure already exists at this position
    #                 stuctures: list[Unit] = [
    #                     s
    #                     for s in self.ai.structures
    #                     if cy_distance_to_squared(s.position, target_pos) < 9.0
    #                 ]
    #
    #                 if len(stuctures) > 0:
    #                     # Structure exists (possibly incomplete), assign SCV to it
    #                     if cy_distance_to_squared(scv.position, target_pos) <= 25.0:
    #                         scv(AbilityId.SMART, stuctures[0])
    #                         self._proxy_tracker[tag]["status"] = ProxySCVStatus.Building
    #                     else:
    #                         scv.move(target_pos)
    #                 elif (
    #                     cy_distance_to_squared(scv.position, target_pos) <= 25.0
    #                     and self.ai.tech_requirement_progress(to_build) >= 1.0
    #                     and self.ai.can_afford(to_build)
    #                 ):
    #                     # No structure exists, build a new one
    #                     scv.build(to_build, target_pos)
    #                     self._proxy_tracker[tag]["status"] = ProxySCVStatus.Building
    #                 else:
    #                     scv.move(target_pos)
    #             case ProxySCVStatus.Building:
    #                 target_pos: Point2 = self._proxy_tracker[tag]["pos"]
    #                 # Check if structure still exists and is incomplete
    #                 stuctures: list[Unit] = [
    #                     s
    #                     for s in self.ai.structures
    #                     if cy_distance_to_squared(s.position, target_pos) < 9.0
    #                 ]
    #
    #                 close_enemy_workers: Units = self.ai.mediator.get_units_in_range(
    #                     start_points=[scv.position],
    #                     distances=9.0,
    #                     query_tree=UnitTreeQueryType.EnemyGround,
    #                 )[0].filter(lambda u: u.type_id in WORKER_TYPES)
    #
    #                 if len(stuctures) == 0:
    #                     # Structure doesn't exist anymore (cancelled or destroyed)
    #                     self._proxy_tracker[tag]["status"] = ProxySCVStatus.Idle
    #                 elif stuctures[0].is_ready:
    #                     # Structure is complete
    #                     self._proxy_tracker[tag]["status"] = ProxySCVStatus.Idle
    #                 elif close_enemy_workers and scv.is_constructing_scv:
    #                     # Enemy nearby and currently building - stop to defend
    #                     scv(AbilityId.HALT)
    #                     self._proxy_tracker[tag]["status"] = ProxySCVStatus.Defending
    #                 elif not scv.is_constructing_scv and not scv.is_moving:
    #                     # SCV is idle and not building - command it to build
    #                     scv(AbilityId.SMART, stuctures[0])
    #             case ProxySCVStatus.Idle:
    #                 self.ai.mediator.assign_role(tag=tag, role=UnitRole.GATHERING)
    #             case ProxySCVStatus.Defending:
    #                 target_pos: Point2 = self._proxy_tracker[tag]["pos"]
    #
    #                 close_enemy_workers: Units = self.ai.mediator.get_units_in_range(
    #                     start_points=[scv.position],
    #                     distances=11.0,
    #                     query_tree=UnitTreeQueryType.EnemyGround,
    #                 )[0].filter(lambda u: u.type_id in WORKER_TYPES)
    #                 if close_enemy_workers:
    #                     scv.attack(close_enemy_workers[0])
    #                 if (
    #                     not close_enemy_workers
    #                     or cy_distance_to_squared(scv.position, target_pos) > 100.0
    #                 ):
    #                     self._proxy_tracker[tag]["status"] = ProxySCVStatus.Moving
