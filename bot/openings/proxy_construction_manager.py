from dataclasses import dataclass
from typing import Optional

from ares import AresBot
from ares.consts import UnitRole, UnitTreeQueryType, WORKER_TYPES, DEBUG
from cython_extensions import cy_distance_to_squared
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.consts import ProxySCVStatus


@dataclass
class BuildTask:
    """Represents a building task at a specific location."""

    structure_type: UnitTypeId
    position: Point2
    assigned_scv_tag: Optional[int] = None
    status: ProxySCVStatus = ProxySCVStatus.Moving
    timestamp: float = 0.0

    def __hash__(self):
        # Hash based on position so we can use it in sets/dicts
        return hash((self.position.x, self.position.y))


class ProxyConstructionManager:
    """Manages proxy building construction with multiple SCVs, handling dead SCVs gracefully."""

    def __init__(self, ai: AresBot):
        self.ai = ai
        # Track build tasks by position (rounded to nearest int for consistency)
        self._build_tasks: dict[tuple[int, int], BuildTask] = {}
        # Track which SCVs are assigned to which tasks
        self._scv_to_task: dict[int, tuple[int, int]] = {}

    def get_position_key(self, pos: Point2) -> tuple[int, int]:
        """Convert position to a hashable key."""
        return (int(pos.x), int(pos.y))

    def _count_structures_near_proxy(
        self,
        proxy_location: Point2,
        structure_type: UnitTypeId,
        search_radius: float = 20.0,
    ) -> int:
        """Count all structures of the given type near the proxy location."""
        count = 0
        for structure in self.ai.structures.filter(
            lambda s: s.type_id == structure_type
        ):
            if (
                cy_distance_to_squared(structure.position, proxy_location)
                < search_radius**2
            ):
                count += 1
        return count

    async def handle_construction(
        self,
        proxy_scvs: Units,
        proxy_location: Point2,
        structure_type: UnitTypeId,
        max_structures: int,
    ) -> None:
        """
        Main method to handle proxy construction.

        Args:
            proxy_scvs: SCVs assigned to proxy building role
            proxy_location: General area where proxy should be built
            structure_type: Type of structure to build
            max_structures: Maximum number of structures to build
        """
        # Step 1: Clean up dead SCVs and find orphaned structures
        self._cleanup_dead_scvs(proxy_scvs)

        # Step 2: Find unfinished structures that need builders
        self._find_orphaned_structures()

        # Step 3: Assign idle SCVs to tasks
        self._assign_idle_scvs_to_tasks(
            proxy_scvs, proxy_location, structure_type, max_structures
        )

        # Step 4: Execute tasks for each SCV
        await self._execute_scv_tasks(proxy_scvs)

    def _cleanup_dead_scvs(self, proxy_scvs: Units) -> None:
        """Remove dead SCVs from task assignments."""
        alive_scv_tags = {scv.tag for scv in proxy_scvs}

        # Find dead SCVs and unassign them from tasks
        dead_scv_tags = []
        for scv_tag, task_key in list(self._scv_to_task.items()):
            if scv_tag not in alive_scv_tags:
                # SCV is dead, unassign from task
                if task_key in self._build_tasks:
                    task = self._build_tasks[task_key]
                    if task.assigned_scv_tag == scv_tag:
                        task.assigned_scv_tag = None
                        task.status = ProxySCVStatus.Moving

                dead_scv_tags.append(scv_tag)

        # Clean up dead SCV mappings
        for scv_tag in dead_scv_tags:
            del self._scv_to_task[scv_tag]

    def _find_orphaned_structures(self) -> None:
        """Find incomplete structures that don't have an assigned builder."""
        for task_key, task in list(self._build_tasks.items()):
            if task.assigned_scv_tag is not None:
                continue  # Task has an assigned SCV

            # Check if there's an incomplete structure at this position
            structures_at_pos = [
                s
                for s in self.ai.structures
                if cy_distance_to_squared(s.position, task.position) < 9.0
            ]

            if structures_at_pos:
                structure = structures_at_pos[0]
                if structure.is_ready:
                    # Structure is complete, remove task
                    del self._build_tasks[task_key]
                    if self.ai.config[DEBUG]:
                        self.ai.client.debug_text_screen(
                            f"Structure at {task.position} completed, removing task",
                            pos=(0.1, 0.35),
                            size=10,
                        )
                # else: structure exists but incomplete, keep task available for assignment
            else:
                # No structure found and no assigned SCV
                # This task might be abandoned (e.g., structure canceled)
                # Keep it in case we want to retry, or remove if too old
                if self.ai.time - task.timestamp > 30.0:
                    del self._build_tasks[task_key]

    def _assign_idle_scvs_to_tasks(
        self,
        proxy_scvs: Units,
        proxy_location: Point2,
        structure_type: UnitTypeId,
        max_structures: int,
    ) -> None:
        """Assign idle SCVs to tasks (prioritize unfinished structures)."""
        # Find idle SCVs (not assigned to any task)
        idle_scvs = [scv for scv in proxy_scvs if scv.tag not in self._scv_to_task]

        if not idle_scvs:
            return

        # First priority: assign to existing tasks without builders
        unassigned_tasks = [
            (task_key, task)
            for task_key, task in self._build_tasks.items()
            if task.assigned_scv_tag is None
        ]

        for task_key, task in unassigned_tasks:
            if not idle_scvs:
                break

            scv = idle_scvs.pop(0)
            task.assigned_scv_tag = scv.tag
            task.status = ProxySCVStatus.Moving
            task.timestamp = self.ai.time
            self._scv_to_task[scv.tag] = task_key

            if self.ai.config[DEBUG]:
                self.ai.client.debug_text_screen(
                    f"Assigned SCV {scv.tag} to existing task at {task.position}",
                    pos=(0.1, 0.4),
                    size=10,
                )

        # Second priority: create new tasks if we haven't reached max structures
        # Count total structures near proxy (both complete and in-progress)
        total_structures = self._count_structures_near_proxy(
            proxy_location, structure_type, search_radius=25.0
        )

        if idle_scvs and total_structures < max_structures:
            for scv in idle_scvs:
                # Recount in case we just created a structure
                total_structures = self._count_structures_near_proxy(
                    proxy_location, structure_type, search_radius=25.0
                )

                if total_structures >= max_structures:
                    break

                # Request a new building placement
                placement = self.ai.mediator.request_building_placement(
                    base_location=proxy_location,
                    structure_type=structure_type,
                    closest_to=self.ai.enemy_start_locations[0],
                )

                if placement:
                    task_key = self.get_position_key(placement)

                    # Make sure we don't already have a task at this position
                    if task_key in self._build_tasks:
                        continue

                    task = BuildTask(
                        structure_type=structure_type,
                        position=placement,
                        assigned_scv_tag=scv.tag,
                        status=ProxySCVStatus.Moving,
                        timestamp=self.ai.time,
                    )

                    self._build_tasks[task_key] = task
                    self._scv_to_task[scv.tag] = task_key

                    if self.ai.config[DEBUG]:
                        self.ai.client.debug_text_screen(
                            f"Created new task at {placement} for SCV {scv.tag}",
                            pos=(0.1, 0.45),
                            size=10,
                        )

    async def _execute_scv_tasks(self, proxy_scvs: Units) -> None:
        """Execute the task for each SCV based on its current status."""
        for scv in proxy_scvs:
            if scv.tag not in self._scv_to_task:
                continue

            task_key = self._scv_to_task[scv.tag]
            if task_key not in self._build_tasks:
                # Task was removed, clean up
                del self._scv_to_task[scv.tag]
                continue

            task = self._build_tasks[task_key]

            # Execute based on current status
            if task.status == ProxySCVStatus.Moving:
                await self._handle_moving(scv, task)
            elif task.status == ProxySCVStatus.Building:
                await self._handle_building(scv, task)
            elif task.status == ProxySCVStatus.Defending:
                await self._handle_defending(scv, task)
            elif task.status == ProxySCVStatus.Idle:
                # Task complete, clean up
                self.ai.mediator.assign_role(tag=scv.tag, role=UnitRole.GATHERING)
                del self._scv_to_task[scv.tag]
                del self._build_tasks[task_key]

    async def _handle_moving(self, scv: Unit, task: BuildTask) -> None:
        """Handle SCV in Moving state."""
        # Check if structure already exists at target
        structures_at_pos = [
            s
            for s in self.ai.structures
            if cy_distance_to_squared(s.position, task.position) < 9.0
        ]

        if structures_at_pos:
            # Structure exists, assign SCV to continue building
            structure = structures_at_pos[0]
            if structure.is_ready:
                task.status = ProxySCVStatus.Idle
            elif cy_distance_to_squared(scv.position, task.position) <= 25.0:
                scv(AbilityId.SMART, structure)
                task.status = ProxySCVStatus.Building
            else:
                scv.move(task.position)
        elif (
            cy_distance_to_squared(scv.position, task.position) <= 25.0
            and self.ai.tech_requirement_progress(task.structure_type) >= 1.0
            and self.ai.can_afford(task.structure_type)
        ):
            # Close enough and can afford, build
            scv.build(task.structure_type, task.position)
            task.status = ProxySCVStatus.Building
        else:
            # Move towards target
            scv.move(task.position)

    async def _handle_building(self, scv: Unit, task: BuildTask) -> None:
        """Handle SCV in Building state."""
        # Check if structure still exists
        structures_at_pos = [
            s
            for s in self.ai.structures
            if cy_distance_to_squared(s.position, task.position) < 9.0
        ]

        if not structures_at_pos:
            # Structure destroyed or doesn't exist
            task.status = ProxySCVStatus.Moving
            return

        structure = structures_at_pos[0]

        if structure.is_ready:
            # Structure complete
            task.status = ProxySCVStatus.Idle
            return

        # Check for nearby enemies
        close_enemy_workers = self.ai.mediator.get_units_in_range(
            start_points=[scv.position],
            distances=9.0,
            query_tree=UnitTreeQueryType.EnemyGround,
        )[0].filter(lambda u: u.type_id in WORKER_TYPES)

        if close_enemy_workers and scv.is_constructing_scv:
            # Enemy nearby while building, defend
            scv(AbilityId.HALT)
            task.status = ProxySCVStatus.Defending
        elif not scv.is_constructing_scv and not scv.is_moving:
            # SCV stopped building but structure incomplete, resume
            scv(AbilityId.SMART, structure)

    async def _handle_defending(self, scv: Unit, task: BuildTask) -> None:
        """Handle SCV in Defending state."""
        close_enemy_workers = self.ai.mediator.get_units_in_range(
            start_points=[scv.position],
            distances=11.0,
            query_tree=UnitTreeQueryType.EnemyGround,
        )[0].filter(lambda u: u.type_id in WORKER_TYPES)

        if cy_distance_to_squared(scv.position, task.position) > 100.0:
            # Too far from build site, return to moving
            task.status = ProxySCVStatus.Moving
        elif close_enemy_workers:
            # Attack nearest enemy worker
            scv.attack(close_enemy_workers[0])
        else:
            # No enemies nearby, return to building
            task.status = ProxySCVStatus.Moving

    def get_num_structures_building(self) -> int:
        """Get the number of structures currently being built."""
        return len(self._build_tasks)

    def is_complete(self, min_structures: int) -> bool:
        """Check if the proxy is complete (has at least min_structures ready)."""
        # Count ready structures near any of our build tasks
        ready_count = 0
        for task_key, task in self._build_tasks.items():
            structures_at_pos = [
                s
                for s in self.ai.structures
                if cy_distance_to_squared(s.position, task.position) < 9.0
                and s.is_ready
            ]
            if structures_at_pos:
                ready_count += 1

        return ready_count >= min_structures
