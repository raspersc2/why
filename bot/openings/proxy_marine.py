from ares import AresBot
from cython_extensions import cy_unit_pending, cy_towards
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from src.ares.consts import UnitRole

from bot.combat.base_combat import BaseCombat
from bot.openings.bio import Bio
from bot.openings.opening_base import OpeningBase


class ProxyMarine(OpeningBase):
    _bio: OpeningBase
    _ground_range_combat: BaseCombat
    _proxy_location: Point2

    def __init__(self):
        super().__init__()
        self._proxy_complete: bool = False

    @property
    def army_comp(self) -> dict:
        return {
            UnitTypeId.MARINE: {"proportion": 1.0, "priority": 0},
        }

    @property
    def upgrade_list(self) -> list[UpgradeId]:
        if len(self.ai.townhalls) < 2:
            return []
        upgrades = [
            UpgradeId.SHIELDWALL,
            UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
            UpgradeId.TERRANINFANTRYARMORSLEVEL1,
        ]
        if self.ai.supply_workers >= 48:
            upgrades.extend(
                [
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL2,
                    UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
                    UpgradeId.TERRANINFANTRYARMORSLEVEL3,
                ]
            )
        return upgrades

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._bio = Bio()
        await self._bio.on_start(ai)
        self._proxy_location = self._calculate_proxy_location()

    async def on_step(self) -> None:
        if self.ai.build_order_runner.build_completed:
            self._macro()

        # Handle proxy construction
        proxy_scvs_amount: int = 3 if not self._proxy_complete else 0
        proxy_scvs = self._handle_proxy_scv_assignment(
            proxy_scvs_amount, self._proxy_location
        )
        if proxy_scvs:
            await self.proxy_construction_manager.handle_construction(
                proxy_scvs, self._proxy_location, UnitTypeId.BARRACKS, 3
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
            >= 3
        ):
            self._proxy_complete = True

        await self._bio.on_step(target=self.attack_target)

    def _macro(self):
        own_rax = self.ai.mediator.get_own_structures_dict[UnitTypeId.BARRACKS]
        self._generic_macro_plan(
            self.army_comp,
            self.ai.start_location,
            self.upgrade_list,
            add_hellions=False,
            add_upgrades=True,
            can_expand=self.ai.minerals > 400,
            freeflow_mode=True,
            upgrade_to_pfs=False,
            num_gas_buildings=0 if len(self.ai.townhalls) < 2 else 1,
            num_one_base_workers=13,
            production_controller_enabled=len(self.ai.townhalls) > 1,
            can_add_orbital=self.ai.minerals >= 200
            or len([r for r in own_rax if r.is_ready and not r.is_active]) == 0,
            add_production_at_bank=(550, 0),
        )

    def on_unit_created(self, unit: Unit) -> None:
        if unit.type_id == UnitTypeId.MARINE:
            self.ai.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
