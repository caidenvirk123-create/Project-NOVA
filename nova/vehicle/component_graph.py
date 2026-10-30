"""
nova.vehicle.component_graph
============================
Hierarchical structure mapping for Project NOVA aerospace systems.

Architecture role — Pipeline Stage 9 (Component Updates)
---------------------------------------------------------
This module provides a Directed Acyclic Graph (DAG) for linking the internal
systems aboard the vehicle (Fuselage sections, Engine Mounts, Payloads, Tanks).

The physics load distributions propagate out across these node graphs down 
`StructuralJoint` links which validate the dynamic axial and torque strains.
It securely binds mathematical representations via the associated MassModel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np

# Directly export properties satisfying architecture integrations matching mass modeling constraints
from nova.vehicle.mass_model import MassComponent, MassModel, compute_mass_properties


@dataclass
class ComponentNode:
    """
    Representation of one distinct structural segment. Connects raw 
    mass physical data with component graph relationship mapping.
    """
    node_id: str
    display_name: str
    mass_component: MassComponent
    component_type: str
    is_separable: bool = False
    is_active: bool = True

    def deactivate(self) -> None:
        """
        Flag this node and its physical mass sub-components as inert. 
        Safely mimics structural separation sequence events logically.
        """
        self.is_active = False
        self.mass_component.is_active = False


@dataclass
class StructuralJoint:
    """
    Structural connection interface between two ComponentNodes. 
    Models physics strength margins calculating Euler buckling and flex yields.
    """
    # Graph linking IDs
    joint_id: str
    parent_id: str
    child_id: str
    
    # Material properties
    material_name: str
    elastic_modulus: float
    area_moment_of_inertia: float
    length: float
    cross_section_area: float
    yield_stress: float
    shear_yield_stress: float
    
    # Dimension estimates
    extreme_fiber_distance: float = 0.01  # Used in M_bend constraints logic (default > 0 to avoid zero-divisions)

    # State Variables updated externally dynamically by the Structural physics solver
    axial_force: float = 0.0
    shear_force: float = 0.0
    bending_moment: float = 0.0
    torsion_moment: float = 0.0
    failed: bool = False

    @property
    def euler_buckling_limit(self) -> float:
        """Calculates critical buckling constraint boundary natively [N]."""
        if self.length <= 0.0:
            return float('inf')
        # P_cr = π²EI / L²
        return (math.pi**2 * self.elastic_modulus * self.area_moment_of_inertia) / (self.length**2)

    @property
    def shear_yield_limit(self) -> float:
        """Rectangular proxy threshold ensuring limits are validated [N]."""
        # V_max = 2/3 * tau_yield * A
        return (2.0 / 3.0) * self.shear_yield_stress * self.cross_section_area

    @property
    def bending_failure_moment(self) -> float:
        """Moments initiating stress fracturing tolerances mapping variables explicitly [N·m]."""
        if self.extreme_fiber_distance <= 0.0:
            return float('inf')
        # M_yield = sigma_yield * I / c
        return (self.yield_stress * self.area_moment_of_inertia) / self.extreme_fiber_distance

    def check_failure(self) -> bool:
        """
        Compares present load boundaries to static structure margins natively. 
        """
        if self.failed:
            return True  # Latch the fracture
            
        if (abs(self.axial_force) > self.euler_buckling_limit or 
            abs(self.shear_force) > self.shear_yield_limit or 
            abs(self.bending_moment) > self.bending_failure_moment):
            self.failed = True
            
        return self.failed


class ComponentGraph:
    """
    DAG handling parent/child interactions mapped throughout spacecraft sequences.
    
    Connects disjoint modules handling stage jettison operations or determining load paths 
    cascading throughout integrated configurations across physics updates effectively.
    Computes exact global mass states on request serving pipeline Stage 3 directly.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ComponentNode] = {}
        self._joints: Dict[str, StructuralJoint] = {}
        
        self._children: Dict[str, List[str]] = {}
        self._parents: Dict[str, List[str]] = {}
        
        # Internal Cache evaluating parameters mathematically dynamically
        self._current_mass_model: Optional[MassModel] = None

    # ------------------------------------------------------------------
    # Data Tracking & Population Integrations
    # ------------------------------------------------------------------

    def add_node(self, node: ComponentNode) -> None:
        """Register structural components allocating relational mapping definitions arrays natively."""
        self._nodes[node.node_id] = node
        if node.node_id not in self._children:
            self._children[node.node_id] = []
            self._parents[node.node_id] = []
        self._current_mass_model = None  # Cache Invalidated

    def add_component(self, component: Union[MassComponent, ComponentNode]) -> None:
        """
        Versatile interface registering hardware directly onto the Graph.
        Permits receiving a primitive MassComponent natively encapsulating it within a node block,
        or resolving native raw `ComponentNode`s identically.
        """
        if isinstance(component, MassComponent):
            # Establish proxy identification securely mimicking nodes structure explicitly 
            comp_node = ComponentNode(
                node_id=component.name,
                display_name=component.name,
                mass_component=component,
                component_type="auto"
            )
            self.add_node(comp_node)
        elif isinstance(component, ComponentNode):
            self.add_node(component)
        else:
            raise TypeError("`add_component` expects type MassComponent or ComponentNode.")

    def add_joint(self, joint: StructuralJoint) -> None:
        """Tie elements ensuring both mapped links precede validation tests structurally securely."""
        if joint.parent_id not in self._nodes or joint.child_id not in self._nodes:
            raise ValueError(
                f"Joint Integration Exception: Mapped structure frames '{joint.parent_id}' "
                f"or '{joint.child_id}' are missing from registered internal components database."
            )
        self._joints[joint.joint_id] = joint
        self._children[joint.parent_id].append(joint.child_id)
        self._parents[joint.child_id].append(joint.parent_id)

    # ------------------------------------------------------------------
    # Mass Physics Aggregation Algorithms (RK4 Compatibility Layer)
    # ------------------------------------------------------------------

    def calculate_mass_properties(self) -> MassModel:
        """
        Actively processes independent properties mathematically scaling offset calculations.
        Executes Parallel-Axis geometric shift processing reliably aggregating parameters.

        Handles degenerate fallback environments maintaining strict integration stability.
        
        Returns:
            MassModel: The frozen summation framework reflecting this timestamp securely.
        """
        active_comps = self.active_mass_components()

        # Handle complete detachment / structural elimination explicitly preserving stable division arithmetic 
        if not active_comps:
            fallback_component = MassComponent(
                name="null_frame",
                mass=1.0e-6,
                position_body=np.zeros(3, dtype=np.float64),
                inertia_own=(1.0e-6 * np.eye(3, dtype=np.float64)),
                is_active=True
            )
            self._current_mass_model = compute_mass_properties([fallback_component])
        else:
            self._current_mass_model = compute_mass_properties(active_comps)
            
        return self._current_mass_model

    # Exposed Physical Proxy Facades for immediate programmatic retrieval access paths

    @property
    def total_mass(self) -> float:
        """Access evaluated absolute mass parameter sum strictly bounding parameters [kg]."""
        if self._current_mass_model is None:
            self.calculate_mass_properties()
        return self._current_mass_model.total_mass

    @property
    def center_of_mass(self) -> np.ndarray:
        """Calculates tracking origins tracking bounds specifically representing frame parameters natively."""
        if self._current_mass_model is None:
            self.calculate_mass_properties()
        return self._current_mass_model.com_body

    @property
    def total_inertia(self) -> np.ndarray:
        """Produces completely compiled full symmetric tensor evaluation mathematically."""
        if self._current_mass_model is None:
            self.calculate_mass_properties()
        return self._current_mass_model.inertia_body


    # ------------------------------------------------------------------
    # Query Fetching 
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> ComponentNode:
        if node_id not in self._nodes:
            raise KeyError(f"Graph Search Fault: Missing expected internal element node {node_id}")
        return self._nodes[node_id]

    def get_component(self, component_id: str) -> Optional[MassComponent]:
        """Fetch physics basis isolated representation natively passing none cleanly mapping values gracefully."""
        node = self._nodes.get(component_id)
        return node.mass_component if node else None

    def get_joint(self, joint_id: str) -> StructuralJoint:
        return self._joints[joint_id]

    def nodes(self) -> List[ComponentNode]:
        """Expose a uniform sequence encompassing both inert & flight-active topology grids natively."""
        return list(self._nodes.values())

    def active_nodes(self) -> List[ComponentNode]:
        """Filter segments isolating exclusively active structural masses sequentially bounded strictly."""
        return [node for node in self._nodes.values() if node.is_active]

    def joints(self) -> List[StructuralJoint]:
        """Provides iterative connectivity reference linking load boundaries explicitly evaluated dynamically."""
        return list(self._joints.values())

    def failed_joints(self) -> List[StructuralJoint]:
        """Determine ruptured linkages returning structure pointers directly validating physics tolerances safely."""
        return [joint for joint in self._joints.values() if joint.failed]

    def children_of(self, node_id: str) -> List[ComponentNode]:
        """Map dependencies tracking immediate descendant physical node blocks attached beneath parents securely."""
        child_ids = self._children.get(node_id, [])
        return [self._nodes[cid] for cid in child_ids]

    def parents_of(self, node_id: str) -> List[ComponentNode]:
        """Traverse upstream fetching dependencies directly maintaining linkage securely bounds testing."""
        parent_ids = self._parents.get(node_id, [])
        return [self._nodes[pid] for pid in parent_ids]

    def active_mass_components(self) -> List[MassComponent]:
        """Extraction layer isolating Mass elements specifically driving integration pipeline mass matrices natively."""
        return [node.mass_component for node in self._nodes.values() if node.is_active]


    # ------------------------------------------------------------------
    # System Status Validations & Graph Traversal Mutators
    # ------------------------------------------------------------------

    def evaluate_structural_failures(self) -> List[str]:
        """
        Calculates and maps failure flags structurally outputting discrete 
        Identifier tracking variables required within system-wide flight termination conditions.
        """
        failed_linkage_ids = []
        for joint in self._joints.values():
            if joint.check_failure():
                failed_linkage_ids.append(joint.joint_id)
        return failed_linkage_ids

    def jettison_subtree(self, node_id: str) -> None:
        """
        Disconnect payload or sub-structures initiating stage detachment.
        Sets 'is_active' inert across the indicated block & all subsequent cascading payload hierarchies reliably.
        """
        eval_stack = [node_id]
        while eval_stack:
            current = eval_stack.pop()
            tgt_node = self._nodes.get(current)
            if tgt_node and tgt_node.is_active:
                tgt_node.deactivate()
            # Descend logic dependencies capturing remaining payload arrays nested mapping logic
            eval_stack.extend(self._children.get(current, []))
            
        self._current_mass_model = None

    def summary(self) -> str:
        """Convenience validation debug tracking variables output."""
        return (
            f"ComponentGraph[Nodes: {len(self.active_nodes())} Active / {len(self.nodes())} Total | "
            f"Joints: {len(self.joints())} Linked ({len(self.failed_joints())} Broken)]"
        )