## Frame2D -solver

import numpy as np
import json
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter, HTMLWriter, PillowWriter
from collections import defaultdict
from scipy.linalg import eig



    
#geometrický uzel
@dataclass
class Node:
    id: int
    x: float
    y: float

#uzel DOF
@dataclass
class DofNode:
    id: int
    node_id: int
    ux: int
    uy: int
    rz: int

    def dof_vector(self):
        return [self.ux, self.uy, self.rz]

#material a prurez
@dataclass
class Section:
    id: int
    E: float
    A: float
    I: float

#prutovy prvek
@dataclass
class Element:
    id: int
    i: int      # ID dof node
    j: int
    section_id: int

#zatížení v uzlu DOF
@dataclass
class NodalLoad:
    dof_node: int
    Fx: float
    Fy: float
    Mz: float

#zatizeni prutu
@dataclass
class ElementLoad:
    element: int
    qx: float
    qz: float


@dataclass
class Mass:
    dof_id: int
    m: float


@dataclass
class DynamicNodalForce:
    dof_id: int
    Re_F: float
    Im_F: float
    Omega: float
    multiplier: int = 1


@dataclass
class DampingRatio:
    mode: int
    zeta: float





## Matematické funkce mimo třídu

#Lokální matice tuhosti
def local_stiffness(E, A, I, L):

    EA = E * A / L
    EI = E * I / L**3

    return np.array([
        [ EA,      0,        0,     -EA,      0,        0],
        [ 0,   12*EI,   6*EI*L,      0, -12*EI,   6*EI*L],
        [ 0,   6*EI*L, 4*EI*L**2,    0, -6*EI*L, 2*EI*L**2],
        [-EA,      0,        0,      EA,      0,        0],
        [ 0,  -12*EI, -6*EI*L,       0,  12*EI, -6*EI*L],
        [ 0,   6*EI*L, 2*EI*L**2,    0, -6*EI*L, 4*EI*L**2]
    ])

#Transformační matice
def transformation_matrix(alpha):

    c = np.cos(alpha)
    s = np.sin(alpha)

    return np.array([
        [ c,  s, 0,  0, 0, 0],
        [-s,  c, 0,  0, 0, 0],
        [ 0,  0, 1,  0, 0, 0],
        [ 0,  0, 0,  c, s, 0],
        [ 0,  0, 0, -s, c, 0],
        [ 0,  0, 0,  0, 0, 1]
    ])


#Matice tuhosti prvku v globalnicm s.s.
def global_stiffness(Kl, alpha):

    T = transformation_matrix(alpha)
    return T.T @ Kl @ T


def global_geometric_stiffness(Kl_sigma, alpha):
    """Transformace geometrické (počáteční napětí) matice z lokálního do globálního s.s."""

    T = transformation_matrix(alpha)
    return T.T @ Kl_sigma @ T


def local_geometric_stiffness(N, L):

    coeff = N / (30 * L)

    return coeff * np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 36, 3*L, 0, -36, 3*L],
        [0, 3*L, 4*L**2, 0, -3*L, -L**2],
        [0, 0, 0, 0, 0, 0],
        [0, -36, -3*L, 0, 36, -3*L],
        [0, 3*L, -L**2, 0, -3*L, 4*L**2],
    ])


#Vektor zatizeni prutu transformovaného do uzlu u lokalnich s.s.
def local_element_load(qx, qz, L):
    """
    Lokální ekvivalentní uzlové síly prutu
    DOF: [u_i, w_i, phi_i, u_j, w_j, phi_j]
    """

    f = np.zeros(6)

    # osové zatížení
    f[0] = qx * L / 2
    f[3] = qx * L / 2

    # příčné zatížení
    f[1] = qz * L / 2
    f[4] = qz * L / 2

    f[2] = qz * L**2 / 12
    f[5] = -qz * L**2 / 12

    return f

#Transformace zatizeni prutu do globalniho s.s.
def global_element_load(qx, qz, L, alpha):
    fl = local_element_load(qx, qz, L)
    T = transformation_matrix(alpha)
    fg = T.T @ fl
    
    return fg

#převod globálních posunů do lokalniho s.s.
def local_displacements(u_global, alpha):

    T = transformation_matrix(alpha)
    return T @ u_global

#lokalni koncove sily z lokalnich posunu
def element_local_forces(E, A, I, L, u_local, qx=0, qz=0):

    Kl = local_stiffness(E, A, I, L)

    f_int = Kl @ u_local

    f_load = local_element_load(qx, qz, L)

    return f_int + f_load

#funkce pro prubehy vnitřních sil po prvku
def element_diagram(x, L, forces, qz=0):

    N1, V1, M1, N2, V2, M2 = forces

    # FEM → diagram konvence
    N1 = -N1
    M1 = -M1


    N = N1 + (N2 - N1) * x / L
    V = V1 + qz * x
    M = M1 + V1 * x + qz * x**2 / 2

    return N, V, M



#Hlavni trida model
class Model:

    def __init__(self):
        self.nodes = {}
        self.dof_nodes = {}
        self.sections = {}
        self.elements = {}
        self.supports = []

        self.nodal_loads = []
        self.element_loads = []
        self.problem_type = "Static"
        self.number_of_eigenvectors = 3
        self.mass = []
        self.dynamic_nodal_forces = []
        self.damping_ratio = []
        self.U = []
        self.dof_map = {}
        self.dynamic_dofs = []
        self.dynamic_active_dofs = []
        self._dynamic_back_substitution = np.array([[]])
        self.eigenvalues = np.array([])
        self.eigenvectors = np.array([[]])
        self.dynamic_force_vector = np.array([], dtype=complex)
        self.dynamic_modal_coordinates = np.array([], dtype=complex)
        self.dynamic_response = np.array([], dtype=complex)
        self.dynamic_harmonic_forces = {}
        self.dynamic_harmonic_modal_coordinates = {}
        self.dynamic_harmonic_responses = {}
        self.dynamic_excitation_frequency = None
        self.frf_harmonic_dof_id = None
        self.frf_max_omega = 0.0
        self.frf_delta_omega = 0.0
        self.frf_omega_values = np.array([])
        self.frf_response_matrix = np.array([[]], dtype=complex)
        self.frf_reduced_dofs = []
        self._last_animation = None
        self._animations = []

    @classmethod
    def from_json(cls, path):
        model = cls()

        with open(path) as f:
            data = json.load(f)

        for n in data["nodes"]:
            model.nodes[n["id"]] = Node(**n)

        for d in data["dof_nodes"]:
            model.dof_nodes[d["id"]] = DofNode(
                id=d["id"],
                node_id=d["node"],
                ux=d["ux"],
                uy=d["uy"],
                rz=d["rz"]
            )

        for s in data["supports"]:
            model.supports.append(s)

        for s in data["sections"]:
            model.sections[s["id"]] = Section(**s)

        for e in data["elements"]:
    
            model.elements[e["id"]] = Element(
                id=e["id"],
                i=e["i"],
                j=e["j"],
                section_id=e["section"],
            )

        for l in data.get("nodal_loads", []):
            model.nodal_loads.append(NodalLoad(**l))

        for l in data.get("element_loads", []):
            model.element_loads.append(ElementLoad(**l))

        model.problem_type = str(data.get("problem_type", "Static")).strip() or "Static"
        model.number_of_eigenvectors = int(data.get("number_of_eigenvectors", 3))
        model.mass = []
        for item in data.get("mass", []):
            try:
                dof_id = int(item.get("dof_id", item.get("dof_node")))
                m_val = float(item.get("m", item.get("mass")))
            except (TypeError, ValueError):
                continue
            model.mass.append(Mass(dof_id=dof_id, m=m_val))
        model.dynamic_nodal_forces = []
        for item in data.get("dynamic_nodal_forces", []):
            try:
                model.dynamic_nodal_forces.append(
                    DynamicNodalForce(
                        dof_id=int(item.get("dof_id", item.get("dof"))),
                        Re_F=float(item.get("Re_F", 0.0)),
                        Im_F=float(item.get("Im_F", 0.0)),
                        Omega=float(item.get("Omega", 0.0)),
                        multiplier=max(1, int(item.get("multiplier", item.get("Multiplier", 1)))),
                    )
                )
            except (TypeError, ValueError):
                continue
        model.damping_ratio = []
        for item in data.get("damping_ratio", []):
            try:
                model.damping_ratio.append(
                    DampingRatio(
                        mode=int(item.get("mode")),
                        zeta=float(item.get("zeta")),
                    )
                )
            except (TypeError, ValueError):
                continue
        frf_dof = data.get("frf_harmonic_dof_id")
        try:
            model.frf_harmonic_dof_id = int(frf_dof) if frf_dof is not None else None
        except (TypeError, ValueError):
            model.frf_harmonic_dof_id = None
        try:
            model.frf_max_omega = float(data.get("frf_max_omega", 0.0))
        except (TypeError, ValueError):
            model.frf_max_omega = 0.0
        try:
            model.frf_delta_omega = float(data.get("frf_delta_omega", 0.0))
        except (TypeError, ValueError):
            model.frf_delta_omega = 0.0

        model.normalize_ids()

        return model

    def normalize_ids(self):
        """Přečísluje ID na souvislé řady (1..n), aby byly konzistentní pro r."""

        if not self.nodes:
            return

        # nodes
        node_id_map = {
            old_id: new_id
            for new_id, old_id in enumerate(sorted(self.nodes.keys()), start=1)
        }
        self.nodes = {
            node_id_map[old_id]: Node(
                id=node_id_map[old_id],
                x=node.x,
                y=node.y,
            )
            for old_id, node in self.nodes.items()
        }

        # sections
        section_id_map = {
            old_id: new_id
            for new_id, old_id in enumerate(sorted(self.sections.keys()), start=1)
        }
        self.sections = {
            section_id_map[old_id]: Section(
                id=section_id_map[old_id],
                E=sec.E,
                A=sec.A,
                I=sec.I,
            )
            for old_id, sec in self.sections.items()
        }

        # dof nodes
        dof_node_id_map = {
            old_id: new_id
            for new_id, old_id in enumerate(sorted(self.dof_nodes.keys()), start=1)
        }

        old_dofs = sorted({
            dof
            for dn in self.dof_nodes.values()
            for dof in (dn.ux, dn.uy, dn.rz)
        })
        dof_id_map = {old_dof: new_dof for new_dof, old_dof in enumerate(old_dofs, start=1)}

        self.dof_nodes = {
            dof_node_id_map[old_id]: DofNode(
                id=dof_node_id_map[old_id],
                node_id=node_id_map[dn.node_id],
                ux=dof_id_map[dn.ux],
                uy=dof_id_map[dn.uy],
                rz=dof_id_map[dn.rz],
            )
            for old_id, dn in self.dof_nodes.items()
        }

        # elements
        element_id_map = {
            old_id: new_id
            for new_id, old_id in enumerate(sorted(self.elements.keys()), start=1)
        }
        self.elements = {
            element_id_map[old_id]: Element(
                id=element_id_map[old_id],
                i=dof_node_id_map[e.i],
                j=dof_node_id_map[e.j],
                section_id=section_id_map[e.section_id],
            )
            for old_id, e in self.elements.items()
        }

        # supports
        self.supports = [
            {
                **sup,
                "node": dof_node_id_map[sup["node"]],
            }
            for sup in self.supports
            if sup.get("node") in dof_node_id_map
        ]

        # nodal loads
        self.nodal_loads = [
            NodalLoad(
                dof_node=dof_node_id_map[load.dof_node],
                Fx=load.Fx,
                Fy=load.Fy,
                Mz=load.Mz,
            )
            for load in self.nodal_loads
            if load.dof_node in dof_node_id_map
        ]

        # element loads
        self.element_loads = [
            ElementLoad(
                element=element_id_map[load.element],
                qx=load.qx,
                qz=load.qz,
            )
            for load in self.element_loads
            if load.element in element_id_map
        ]

        # lumped masses
        self.mass = [
            Mass(dof_id=dof_id_map[item.dof_id], m=float(item.m))
            for item in self.mass
            if item.dof_id in dof_id_map
        ]
        self.dynamic_nodal_forces = [
            DynamicNodalForce(
                dof_id=dof_id_map[item.dof_id],
                Re_F=float(item.Re_F),
                Im_F=float(item.Im_F),
                Omega=float(item.Omega),
                multiplier=max(1, int(getattr(item, "multiplier", 1))),
            )
            for item in self.dynamic_nodal_forces
            if item.dof_id in dof_id_map
        ]
        self.damping_ratio = [
            DampingRatio(mode=int(item.mode), zeta=float(item.zeta))
            for item in self.damping_ratio
            if int(item.mode) >= 1
        ]

    def to_json_data(self):
        return {
            "nodes": [
                {"id": n.id, "x": n.x, "y": n.y}
                for n in sorted(self.nodes.values(), key=lambda n: n.id)
            ],
            "dof_nodes": [
                {
                    "id": d.id,
                    "node": d.node_id,
                    "ux": d.ux,
                    "uy": d.uy,
                    "rz": d.rz,
                }
                for d in sorted(self.dof_nodes.values(), key=lambda d: d.id)
            ],
            "sections": [
                {"id": s.id, "E": s.E, "A": s.A, "I": s.I}
                for s in sorted(self.sections.values(), key=lambda s: s.id)
            ],
            "elements": [
                {"id": e.id, "i": e.i, "j": e.j, "section": e.section_id}
                for e in sorted(self.elements.values(), key=lambda e: e.id)
            ],
            "supports": sorted(self.supports, key=lambda s: s.get("node", 0)),
            "nodal_loads": [
                {"dof_node": l.dof_node, "Fx": l.Fx, "Fy": l.Fy, "Mz": l.Mz}
                for l in sorted(self.nodal_loads, key=lambda l: l.dof_node)
            ],
            "element_loads": [
                {"element": l.element, "qx": l.qx, "qz": l.qz}
                for l in sorted(self.element_loads, key=lambda l: l.element)
            ],
            "problem_type": self.problem_type,
            "number_of_eigenvectors": self.number_of_eigenvectors,
            "mass": [
                {"dof_id": m.dof_id, "mass": m.m}
                for m in sorted(self.mass, key=lambda item: item.dof_id)
            ],
            "dynamic_nodal_forces": [
                {
                    "dof_id": item.dof_id,
                    "Re_F": item.Re_F,
                    "Im_F": item.Im_F,
                    "Omega": item.Omega,
                    "multiplier": getattr(item, "multiplier", 1),
                }
                for item in sorted(self.dynamic_nodal_forces, key=lambda val: (val.dof_id, val.Omega, getattr(val, "multiplier", 1)))
            ],
            "damping_ratio": [
                {"mode": item.mode, "zeta": item.zeta}
                for item in sorted(self.damping_ratio, key=lambda val: val.mode)
            ],
            "frf_harmonic_dof_id": self.frf_harmonic_dof_id,
            "frf_max_omega": self.frf_max_omega,
            "frf_delta_omega": self.frf_delta_omega,
        }


    #Fixni DOF
    def get_fixed_dofs(self):
        fixed = set()

        for s in self.supports:
            node = s["node"]
            dn = self.dof_nodes[node]

            if s.get("ux"):
                fixed.add(dn.ux)

            if s.get("uy"):
                fixed.add(dn.uy)

            if s.get("rz"):
                fixed.add(dn.rz)

        return fixed
    

    #Helper funkce – ID pole prvku - kodova cisla prutoveho prvku
    def element_dof_ids(self, element_id):
        e = self.elements[element_id]

        di = self.dof_nodes[e.i]
        dj = self.dof_nodes[e.j]

        return [
            di.ux, di.uy, di.rz,
            dj.ux, dj.uy, dj.rz
        ]

    #Počet globálních DOF
    def ndof(self):
        max_id = 0

        for d in self.dof_nodes.values():
            max_id = max(max_id, d.ux, d.uy, d.rz)

        return max_id

    #Funkce - geometrie prvku - dolka a uhel alpha
    def element_geometry(self, element_id):
        e = self.elements[element_id]

        di = self.dof_nodes[e.i]
        dj = self.dof_nodes[e.j]

        ni = self.nodes[di.node_id]
        nj = self.nodes[dj.node_id]

        dx = nj.x - ni.x
        dy = nj.y - ni.y

        L = np.sqrt(dx**2 + dy**2)
        alpha = np.arctan2(dy, dx)

        return L, alpha

    #sestaveni matice tuhosti prvku v globalnim s.s.
    def element_global_stiffness(self, elem):

        # geometrie
        L, alpha = self.element_geometry(elem.id)

        # materiál
        sec = self.sections[elem.section_id]

        # lokální matice
        Kl = local_stiffness(sec.E, sec.A, sec.I, L)

 
        # globální matice
        Kg = global_stiffness(Kl, alpha)

        return Kg


    #Funkce - matice tuhosti globalniho modelu - lokalizace dle DOF
    def assemble_global_stiffness(self):

        K = np.zeros((self.ndof, self.ndof))

        for elem in self.elements.values():

            # lokální matice tuhosti v globálu
            Kg = self.element_global_stiffness(elem)

            # globální DOF indexy prvku
            dof_ids = self.element_dof_ids(elem.id)

            # sestavení
            for i in range(6):
                I = dof_ids[i]

                if I not in self.dof_map:
                    continue

                ii = self.dof_map[I]

                for j in range(6):
                    J = dof_ids[j]

                    if J not in self.dof_map:
                        continue

                    jj = self.dof_map[J]

                    K[ii, jj] += Kg[i, j]

        return K

    #Uzlove zatizeni - vektor pravych stran f - globalni model - lokalizace
    def assemble_nodal_loads(self, f):
        for load in self.nodal_loads:

            d = self.dof_nodes[load.dof_node]

            dofs = [d.ux, d.uy, d.rz]
            forces = [load.Fx, load.Fy, load.Mz]

            for dof, val in zip(dofs, forces):
                if dof > 0:
                    f[dof-1] += val


    #zatizeni prutu - vektor pravych stran - globalni model - lokalizace
    def assemble_element_loads(self):

        loads = {}

        for eload in self.element_loads:

            elem = self.elements[eload.element]

            L, alpha = self.element_geometry(elem.id)
            Fe_global = global_element_load(eload.qx, eload.qz, L, alpha)

            dof_ids = self.element_dof_ids(elem.id)

            for i in range(6):
                dof = dof_ids[i]
                loads[dof] = loads.get(dof, 0.0) + Fe_global[i]

        return loads
    
    #sestavení vektoru prave strany  f  globalni model - lokalizace
    def assemble_global_load_vector(self):

        import numpy as np

        F = np.zeros(self.ndof)

        # nodální zatížení
        for load in self.nodal_loads:

            dn = self.dof_nodes[load.dof_node]

            dofs = [
                (dn.ux, load.Fx),
                (dn.uy, load.Fy),
                (dn.rz, load.Mz)
            ]

            for dof, value in dofs:
                if dof in self.dof_map:
                    F[self.dof_map[dof]] += value

        # zatížení na prvcích
        Fe = self.assemble_element_loads()

        for dof, value in Fe.items():
            if dof in self.dof_map:
                F[self.dof_map[dof]] += value

        return F


    #statická kondenzace K a f
    def apply_supports(self, K, F):

        fixed = self.get_fixed_dofs()

        free = [i for i in range(len(F)) if i not in fixed]

        K_red = K[np.ix_(free, free)]
        F_red = F[free]

        return K_red, F_red, free

    #řešení soustavy K r = f
    def solve(self):

        self.initialize_active_dofs()

        # ️sestavení K a f
        K = self.assemble_global_stiffness()
        f = self.assemble_global_load_vector()

        # ️řešení
        self.U = np.linalg.solve(K, f)

    def _dynamic_mass_dofs(self):
        mass_dofs = []
        for item in self.mass:
            if item.dof_id in self.dof_map and item.m > 0:
                mass_dofs.append(item.dof_id)

        return sorted(set(mass_dofs))

    def _condense_dynamic_stiffness(self, K):
        self.dynamic_dofs = self._dynamic_mass_dofs()
        self.dynamic_active_dofs = sorted(self.dof_map, key=self.dof_map.get)
        self._dynamic_back_substitution = np.array([[]])

        if not self.dynamic_dofs:
            raise ValueError("Dynamic analysis requires at least one active DOF with positive mass.")

        master_idx = [self.dof_map[dof] for dof in self.dynamic_dofs]
        slave_idx = [
            idx for idx, dof in enumerate(self.dynamic_active_dofs)
            if dof not in self.dynamic_dofs
        ]

        if not slave_idx:
            K_red = K[np.ix_(master_idx, master_idx)]
            return K_red, self.dynamic_dofs

        K_mm = K[np.ix_(master_idx, master_idx)]
        K_ms = K[np.ix_(master_idx, slave_idx)]
        K_sm = K[np.ix_(slave_idx, master_idx)]
        K_ss = K[np.ix_(slave_idx, slave_idx)]

        K_ss_inv_K_sm = np.linalg.solve(K_ss, K_sm)
        K_red = K_mm - K_ms @ K_ss_inv_K_sm
        self._dynamic_back_substitution = -K_ss_inv_K_sm

        return K_red, self.dynamic_dofs

    def assemble_mass_matrix(self):
        if not self.dynamic_dofs:
            raise ValueError("Dynamic DOFs are not initialized. Condense stiffness first.")

        M = np.zeros((len(self.dynamic_dofs), len(self.dynamic_dofs)))
        mass_lookup = {item.dof_id: item.m for item in self.mass if item.m > 0}

        for i, dof in enumerate(self.dynamic_dofs):
            M[i, i] = mass_lookup.get(dof, 0.0)

        if np.any(np.diag(M) <= 0):
            raise ValueError("All reduced dynamic DOFs must have positive mass on M diagonal.")

        return M

    def solve_dynamic(self):
        self.initialize_active_dofs()
        K = self.assemble_global_stiffness()
        K_red, reduced_dofs = self._condense_dynamic_stiffness(K)
        M = self.assemble_mass_matrix()

        A = np.linalg.solve(M, K_red)
        eigvals, eigvecs = np.linalg.eig(A)

        order = np.argsort(eigvals.real)
        eigvals = eigvals[order].real
        eigvecs = eigvecs[:, order].real

        n = min(self.number_of_eigenvectors, len(eigvals))
        self.eigenvalues = eigvals[:n]
        self.eigenvectors = eigvecs[:, :n]
        self.dynamic_force_vector = np.zeros(len(reduced_dofs), dtype=complex)
        self.dynamic_modal_coordinates = np.zeros(n, dtype=complex)
        self.dynamic_response = np.zeros(len(reduced_dofs), dtype=complex)
        self.dynamic_harmonic_forces = {}
        self.dynamic_harmonic_modal_coordinates = {}
        self.dynamic_harmonic_responses = {}
        self.dynamic_excitation_frequency = None

        return self.eigenvalues, self.eigenvectors, reduced_dofs

    def solve_dynamic_steady_state(self):
        eigvals, eigvecs, reduced_dofs = self.solve_dynamic()

        if len(self.dynamic_nodal_forces) == 0:
            raise ValueError("Dynamic - steady state requires at least one DynamicNodalForce entry.")

        omega_values = {float(item.Omega) for item in self.dynamic_nodal_forces}
        if len(omega_values) != 1:
            raise ValueError("Dynamic - steady state requires one common base Omega for all DynamicNodalForce entries.")

        omega_base = omega_values.pop()
        reduced_dof_set = set(reduced_dofs)
        invalid_dofs = sorted({item.dof_id for item in self.dynamic_nodal_forces if item.dof_id not in reduced_dof_set})
        if invalid_dofs:
            raise ValueError(
                "DynamicNodalForce dof_id must be active in reduced stiffness matrix K_red. "
                f"Invalid dof_id values: {invalid_dofs}; active reduced DOFs: {reduced_dofs}"
            )

        damping_lookup = {item.mode: float(item.zeta) for item in self.damping_ratio}
        harmonic_forces = {}
        for item in self.dynamic_nodal_forces:
            multiplier = max(1, int(getattr(item, "multiplier", 1)))
            f_h = harmonic_forces.setdefault(multiplier, np.zeros(len(reduced_dofs), dtype=complex))
            idx = reduced_dofs.index(item.dof_id)
            f_h[idx] += complex(item.Re_F, item.Im_F)

        harmonic_modal_coordinates = {}
        harmonic_responses = {}
        total_response = np.zeros(len(reduced_dofs), dtype=complex)
        total_force = np.zeros(len(reduced_dofs), dtype=complex)

        for multiplier, f_h in sorted(harmonic_forces.items()):
            omega_exc = multiplier * omega_base
            q_h = np.zeros(len(eigvals), dtype=complex)

            for mode_idx in range(len(eigvals)):
                phi_i = eigvecs[:, mode_idx]
                omega_i = np.sqrt(max(eigvals[mode_idx], 0.0))
                zeta_i = damping_lookup.get(mode_idx + 1, 0.0)
                f_norm = np.dot(phi_i.T, f_h)
                denominator = (omega_i ** 2 - omega_exc ** 2) + 2j * zeta_i * omega_i * omega_exc
                if abs(denominator) < 1e-12:
                    raise ValueError(
                        f"Steady-state denominator is zero for mode {mode_idx + 1} and multiplier {multiplier}. "
                        "Adjust damping ratio or excitation frequency."
                    )
                q_h[mode_idx] = f_norm / denominator

            u_h = eigvecs @ q_h
            harmonic_modal_coordinates[multiplier] = q_h
            harmonic_responses[multiplier] = u_h
            total_response += u_h
            total_force += f_h

        self.dynamic_force_vector = total_force
        self.dynamic_modal_coordinates = np.sum(np.array(list(harmonic_modal_coordinates.values())), axis=0) if harmonic_modal_coordinates else np.zeros(len(eigvals), dtype=complex)
        self.dynamic_response = total_response
        self.dynamic_harmonic_forces = harmonic_forces
        self.dynamic_harmonic_modal_coordinates = harmonic_modal_coordinates
        self.dynamic_harmonic_responses = harmonic_responses
        self.dynamic_excitation_frequency = omega_base

        return total_response, total_force, reduced_dofs

    def solve_dynamic_frf(self):
        eigvals, eigvecs, reduced_dofs = self.solve_dynamic()

        if self.frf_harmonic_dof_id is None:
            raise ValueError("Dynamic - FRF requires 'frf_harmonic_dof_id'.")
        if self.frf_harmonic_dof_id not in reduced_dofs:
            raise ValueError(
                f"FRF harmonic dof_id {self.frf_harmonic_dof_id} must be one of reduced DOFs: {reduced_dofs}"
            )
        if self.frf_max_omega < 0:
            raise ValueError("frf_max_omega must be >= 0.")
        if self.frf_delta_omega <= 0:
            raise ValueError("frf_delta_omega must be > 0.")

        damping_lookup = {item.mode: float(item.zeta) for item in self.damping_ratio}
        omega_values = np.arange(0.0, self.frf_max_omega + 0.5 * self.frf_delta_omega, self.frf_delta_omega, dtype=float)
        if omega_values.size == 0:
            omega_values = np.array([0.0], dtype=float)

        force = np.zeros(len(reduced_dofs), dtype=complex)
        force[reduced_dofs.index(self.frf_harmonic_dof_id)] = 1.0 + 0.0j

        frf_response = np.zeros((omega_values.size, len(reduced_dofs)), dtype=complex)
        for omega_idx, omega_exc in enumerate(omega_values):
            q_omega = np.zeros(len(eigvals), dtype=complex)
            for mode_idx in range(len(eigvals)):
                phi_i = eigvecs[:, mode_idx]
                omega_i = np.sqrt(max(eigvals[mode_idx], 0.0))
                zeta_i = damping_lookup.get(mode_idx + 1, 0.0)
                f_norm = np.dot(phi_i.T, force)
                denominator = (omega_i ** 2 - omega_exc ** 2) + 2j * zeta_i * omega_i * omega_exc
                if abs(denominator) < 1e-12:
                    denominator = denominator + 1e-12j
                q_omega[mode_idx] = f_norm / denominator
            frf_response[omega_idx, :] = eigvecs @ q_omega

        self.frf_omega_values = omega_values
        self.frf_response_matrix = frf_response
        self.frf_reduced_dofs = list(reduced_dofs)
        self.dynamic_force_vector = force
        self.dynamic_response = frf_response[-1, :]
        self.dynamic_modal_coordinates = np.zeros(len(eigvals), dtype=complex)
        self.dynamic_excitation_frequency = None

        return omega_values, frf_response, reduced_dofs

    def plot_dynamic_frf(self, dof_ids=None, show=False):
        if self.frf_omega_values.size == 0 or self.frf_response_matrix.size == 0:
            raise ValueError("No FRF results available. Run solve_dynamic_frf() first.")

        if dof_ids is None:
            dof_ids = self.frf_reduced_dofs
        dof_ids = [int(d) for d in dof_ids if int(d) in self.frf_reduced_dofs]
        if not dof_ids:
            raise ValueError("No valid dof_ids selected for FRF plot.")

        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        ax_abs, ax_phase = axes
        fig_mobility, axes_mobility = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        ax_abs_mobility, ax_phase_mobility = axes_mobility
        fig_accelerance, axes_accelerance = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
        ax_abs_accelerance, ax_phase_accelerance = axes_accelerance

        for dof_id in dof_ids:
            idx = self.frf_reduced_dofs.index(dof_id)
            U_i = self.frf_response_matrix[:, idx]
            freq_hz = self.frf_omega_values / (2 * np.pi)
            mobility_i = 1j * self.frf_omega_values * U_i
            accelerance_i = -(self.frf_omega_values ** 2) * U_i

            magnitude_db = 20.0 * np.log10(np.maximum(np.abs(U_i), 1e-30))
            ax_abs.plot(freq_hz, magnitude_db, label=f"dof {dof_id}")
            ax_phase.plot(freq_hz, np.arctan2(np.imag(U_i), np.real(U_i)), label=f"dof {dof_id}")

            magnitude_mobility_db = 20.0 * np.log10(np.maximum(np.abs(mobility_i), 1e-30))
            ax_abs_mobility.plot(freq_hz, magnitude_mobility_db, label=f"dof {dof_id}")
            ax_phase_mobility.plot(
                freq_hz, np.arctan2(np.imag(mobility_i), np.real(mobility_i)), label=f"dof {dof_id}"
            )

            magnitude_accelerance_db = 20.0 * np.log10(np.maximum(np.abs(accelerance_i), 1e-30))
            ax_abs_accelerance.plot(freq_hz, magnitude_accelerance_db, label=f"dof {dof_id}")
            ax_phase_accelerance.plot(
                freq_hz, np.arctan2(np.imag(accelerance_i), np.real(accelerance_i)), label=f"dof {dof_id}"
            )

        ax_abs.set_ylabel(r"|U_i| [dB, ref. 1]")
        ax_abs.grid(True, linestyle="--", alpha=0.4)
        ax_abs.legend(loc="best")
        ax_abs.set_title(r"FRF (F = cos(2\pi f t), unit amplitude)")

        ax_phase.set_xlabel("f [Hz]")
        ax_phase.set_ylabel("phase [rad]")
        ax_phase.grid(True, linestyle="--", alpha=0.4)
        ax_phase.legend(loc="best")

        ax_abs_mobility.set_ylabel(r"|i\omega U_i| [dB, ref. 1]")
        ax_abs_mobility.grid(True, linestyle="--", alpha=0.4)
        ax_abs_mobility.legend(loc="best")
        ax_abs_mobility.set_title(r"Mobility FRF ($i\omega \cdot$ Compliance)")

        ax_phase_mobility.set_xlabel("f [Hz]")
        ax_phase_mobility.set_ylabel("phase [rad]")
        ax_phase_mobility.grid(True, linestyle="--", alpha=0.4)
        ax_phase_mobility.legend(loc="best")

        ax_abs_accelerance.set_ylabel(r"|-\omega^2 U_i| [dB, ref. 1]")
        ax_abs_accelerance.grid(True, linestyle="--", alpha=0.4)
        ax_abs_accelerance.legend(loc="best")
        ax_abs_accelerance.set_title(r"Accelerance FRF ($-\omega^2 \cdot$ Compliance)")

        ax_phase_accelerance.set_xlabel("f [Hz]")
        ax_phase_accelerance.set_ylabel("phase [rad]")
        ax_phase_accelerance.grid(True, linestyle="--", alpha=0.4)
        ax_phase_accelerance.legend(loc="best")

        fig.tight_layout()
        fig_mobility.tight_layout()
        fig_accelerance.tight_layout()

        if show:
            plt.show()
        return fig, axes


    def _validate_stability_qx(self):
        invalid_elements = [load.element for load in self.element_loads if abs(load.qx) > 1e-12]
        if invalid_elements:
            ids = sorted(set(invalid_elements))
            raise ValueError(
                "Stability analysis requires qx = 0 on all elements (constant N per element). "
                f"Invalid element ids: {ids}"
            )

    def _get_full_displacement_vector(self):
        max_dof = max(max(dn.ux, dn.uy, dn.rz) for dn in self.dof_nodes.values())
        U_full = np.zeros(max_dof)
        for dof, idx in self.dof_map.items():
            U_full[dof - 1] = self.U[idx]
        return U_full

    def _element_axial_force(self, elem, U_full):
        sec = self.sections[elem.section_id]
        L, alpha = self.element_geometry(elem.id)
        dof_ids = self.element_dof_ids(elem.id)
        u_global = np.array([U_full[dof - 1] for dof in dof_ids])
        u_local = local_displacements(u_global, alpha)

        qx, qz = self.get_element_loads(elem)
        f_local = element_local_forces(sec.E, sec.A, sec.I, L, u_local, qx=qx, qz=qz)

        n_start = -f_local[0]
        n_end = f_local[3]
        return 0.5 * (n_start + n_end)

    def assemble_global_geometric_stiffness(self, axial_forces):
        K_sigma = np.zeros((self.ndof, self.ndof))

        for elem in self.elements.values():
            L, alpha = self.element_geometry(elem.id)
            N = axial_forces[elem.id]

            Kl_sigma = local_geometric_stiffness(N, L)
            Kg_sigma = global_geometric_stiffness(Kl_sigma, alpha)

            dof_ids = self.element_dof_ids(elem.id)

            for i in range(6):
                I = dof_ids[i]
                if I not in self.dof_map:
                    continue
                ii = self.dof_map[I]

                for j in range(6):
                    J = dof_ids[j]
                    if J not in self.dof_map:
                        continue
                    jj = self.dof_map[J]
                    K_sigma[ii, jj] += Kg_sigma[i, j]

        return K_sigma

    def solve_stability(self):
        self._validate_stability_qx()

        self.solve()

        axial_forces = self.get_stability_axial_forces()

        K = self.assemble_global_stiffness()
        K_sigma = self.assemble_global_geometric_stiffness(axial_forces)

        eigvals, eigvecs = eig(K, -K_sigma) # Matice K_sigma může být singulární, proto pozor na použitý algoritmus výpočtu vlastního problému

        order = np.argsort(eigvals.real)
        eigvals = eigvals[order].real
        eigvecs = eigvecs[:, order].real

        self.eigenvalues = eigvals
        self.eigenvectors = eigvecs

        return eigvals, eigvecs

    def get_stability_axial_forces(self):
        """Vrátí osové síly prvků pro sestavení geometrické matice tuhosti."""
        U_full = self._get_full_displacement_vector()
        axial_forces = {}
        for elem in self.elements.values():
            axial_forces[elem.id] = self._element_axial_force(elem, U_full)
        return axial_forces

    def format_stability_results(self):
        if len(self.eigenvalues) == 0:
            return "No stability results available."

        lines = ["STABILITY EIGEN SOLUTION:"]
        active_dofs = sorted(self.dof_map, key=self.dof_map.get)
        lines.append(f"DOFs = {active_dofs}")
        for i, lam in enumerate(self.eigenvalues[:self.number_of_eigenvectors], start=1):
            lines.append(f"alpha_cr_{i} = {lam:.6e}")
            lines.append(f"U_{i} = {self.eigenvectors[:, i - 1]}")

        return "\n".join(lines)

    def print_stability_results(self):
        print(self.format_stability_results())

    def _stability_shape_full_vector(self, mode_index):
        if len(self.eigenvalues) == 0:
            raise ValueError("No stability results available. Run solve_stability() first.")
        if mode_index < 0 or mode_index >= self.eigenvectors.shape[1]:
            raise IndexError("Mode index out of range.")

        max_dof = max(max(dn.ux, dn.uy, dn.rz) for dn in self.dof_nodes.values())
        U_mode = np.zeros(max_dof)

        mode_vec = self.eigenvectors[:, mode_index]
        for dof, idx in self.dof_map.items():
            U_mode[dof - 1] = mode_vec[idx]

        return U_mode

    def plot_stability_shape(self, mode_index, scale=None, n_points=20, show=False):
        U_mode = self._stability_shape_full_vector(mode_index)

        if scale is None:
            max_disp = np.max(np.abs(U_mode))
            if max_disp == 0:
                scale = 1.0
            else:
                size = max(
                    max(n.x for n in self.nodes.values()) - min(n.x for n in self.nodes.values()),
                    max(n.y for n in self.nodes.values()) - min(n.y for n in self.nodes.values()),
                )
                scale = 0.2 * size / max_disp

        fig, ax = plt.subplots()
        plotted_x = []
        plotted_y = []
        plotted_x = []
        plotted_y = []

        for elem in self.elements.values():
            di = self.dof_nodes[elem.i]
            dj = self.dof_nodes[elem.j]
            ni = self.nodes[di.node_id]
            nj = self.nodes[dj.node_id]
            ax.plot([ni.x, nj.x], [ni.y, nj.y], 'k--', linewidth=1)

        self.plot_releases(ax)
        self.plot_supports(ax)

        for elem in self.elements.values():
            i, j = elem.i, elem.j
            L, alpha = self.element_geometry(elem.id)
            c = np.cos(alpha)
            s = np.sin(alpha)

            di = self.dof_nodes[i]
            dj = self.dof_nodes[j]
            ni = self.nodes[di.node_id]
            xi, yi = ni.x, ni.y

            u_global = np.array([
                U_mode[di.ux - 1],
                U_mode[di.uy - 1],
                U_mode[di.rz - 1],
                U_mode[dj.ux - 1],
                U_mode[dj.uy - 1],
                U_mode[dj.rz - 1],
            ])

            T = np.array([
                [c, s, 0, 0, 0, 0],
                [-s, c, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [0, 0, 0, 0, 0, 1],
            ])
            u_local = T @ u_global

            v1, phi1, v2, phi2 = u_local[1], u_local[2], u_local[4], u_local[5]
            u1, u2 = u_local[0], u_local[3]

            xs = []
            ys = []
            for xi_loc in np.linspace(0, 1, n_points):
                N1 = 1 - 3 * xi_loc ** 2 + 2 * xi_loc ** 3
                N2 = L * (xi_loc - 2 * xi_loc ** 2 + xi_loc ** 3)
                N3 = 3 * xi_loc ** 2 - 2 * xi_loc ** 3
                N4 = L * (-xi_loc ** 2 + xi_loc ** 3)

                v = N1 * v1 + N2 * phi1 + N3 * v2 + N4 * phi2
                x_local = xi_loc * L
                u_axial = (1 - xi_loc) * u1 + xi_loc * u2

                xg = xi + c * (x_local + scale * u_axial) - s * (scale * v)
                yg = yi + s * (x_local + scale * u_axial) + c * (scale * v)
                xs.append(xg)
                ys.append(yg)

            ax.plot(xs, ys, 'r', linewidth=2)

        lam = self.eigenvalues[mode_index]
        ax.set_title(fr"Eigen shape no. {mode_index+1}: $\alpha_{{cr}}$ = {lam:.2f}")
        ax.axis("equal")
        ax.set_axis_off()

        if show:
            plt.show()

    def plot_all_stability_shapes(self, show=False):
        n_modes = min(self.number_of_eigenvectors, self.eigenvectors.shape[1] if self.eigenvectors.ndim == 2 else 0)
        for i in range(n_modes):
            self.plot_stability_shape(i, show=False)
        if show:
            self.show_all_plots()

    def _mode_shape_full_vector(self, mode_index):
        if len(self.eigenvalues) == 0:
            raise ValueError("No dynamic results available. Run solve_dynamic() first.")
        if mode_index < 0 or mode_index >= self.eigenvectors.shape[1]:
            raise IndexError("Mode index out of range.")

        max_dof = max(
            max(dn.ux, dn.uy, dn.rz)
            for dn in self.dof_nodes.values()
        )
        U_mode = np.zeros(max_dof)

        mode_vec = self.eigenvectors[:, mode_index]

        if self._dynamic_back_substitution.size:
            slave_modes = self._dynamic_back_substitution @ mode_vec
            slave_dofs = [
                dof for dof in self.dynamic_active_dofs
                if dof not in self.dynamic_dofs
            ]
            for i, dof in enumerate(slave_dofs):
                U_mode[dof - 1] = slave_modes[i]

        for i, dof in enumerate(self.dynamic_dofs):
            U_mode[dof - 1] = mode_vec[i]

        return U_mode

    def _dynamic_response_full_vector(self, response=None):
        if response is None:
            response = self.dynamic_response
        response = np.asarray(response, dtype=complex)
        if response.size == 0:
            raise ValueError("No dynamic steady-state response available.")

        max_dof = max(
            max(dn.ux, dn.uy, dn.rz)
            for dn in self.dof_nodes.values()
        )
        U_full = np.zeros(max_dof, dtype=complex)

        if self._dynamic_back_substitution.size:
            slave_response = self._dynamic_back_substitution @ response
            slave_dofs = [
                dof for dof in self.dynamic_active_dofs
                if dof not in self.dynamic_dofs
            ]
            for i, dof in enumerate(slave_dofs):
                U_full[dof - 1] = slave_response[i]

        for i, dof in enumerate(self.dynamic_dofs):
            U_full[dof - 1] = response[i]

        return U_full

    def format_dynamic_results(self):
        if len(self.eigenvalues) == 0:
            return "No dynamic results available."

        lines = ["DYNAMIC EIGEN SOLUTION:"]
        lines.append(f"Reduced DOFs: {self.dynamic_dofs}")

        for i, lam in enumerate(self.eigenvalues, start=1):
            lines.append(f"omega^2_{i} = {lam:.6e}")
            mode = self.eigenvectors[:, i - 1]
            lines.append(f"U_{i} = {mode}")

        if self.dynamic_excitation_frequency is not None:
            lines.append("")
            lines.append("DYNAMIC STEADY-STATE SOLUTION:")
            lines.append(f"Base Omega = {self.dynamic_excitation_frequency:.6e}")
            if self.dynamic_harmonic_forces:
                for multiplier in sorted(self.dynamic_harmonic_forces):
                    omega_h = multiplier * self.dynamic_excitation_frequency
                    lines.append(f"harmonic {multiplier} * Omega = {omega_h:.6e}")
                    lines.append(f"f[{multiplier}] = {self.dynamic_harmonic_forces[multiplier]}")
                    for i, q_i in enumerate(self.dynamic_harmonic_modal_coordinates.get(multiplier, []), start=1):
                        lines.append(f"q[{multiplier}]_{i} = {q_i}")
                    lines.append(f"u[{multiplier}] = {self.dynamic_harmonic_responses.get(multiplier)}")
            else:
                lines.append(f"f = {self.dynamic_force_vector}")
                for i, q_i in enumerate(self.dynamic_modal_coordinates, start=1):
                    lines.append(f"q_{i} = {q_i}")
                lines.append(f"u = {self.dynamic_response}")
        elif self.frf_omega_values.size > 0 and self.frf_response_matrix.size > 0:
            lines.append("")
            lines.append("DYNAMIC FRF SOLUTION:")
            lines.append(f"harmonic load dof_id = {self.frf_harmonic_dof_id}")
            lines.append(f"omega range: 0 .. {self.frf_max_omega:.6e} (step {self.frf_delta_omega:.6e})")
            lines.append(f"number of FRF points = {self.frf_omega_values.size}")

        return "\n".join(lines)

    def print_dynamic_results(self):
        print(self.format_dynamic_results())

    def initialize_active_dofs(self):

        fixed = self.get_fixed_dofs()

        # aktivní DOF = všechny - fixní
        all_dofs = sorted({
            dn.ux for dn in self.dof_nodes.values()
        } | {
            dn.uy for dn in self.dof_nodes.values()
        } | {
            dn.rz for dn in self.dof_nodes.values()
        })

        active = [d for d in all_dofs if d not in fixed]

        self.dof_map = {d: i for i, d in enumerate(active)}
        self.ndof = len(active)

    def print_global_stiffness_with_dofs(self):
        self.initialize_active_dofs()
        K = self.assemble_global_stiffness()

        print("Global stiffness matrix K:")
        print(K)
        print("\nDOF mapping for K rows/columns:")

        ordered_dofs = sorted(self.dof_map.items(), key=lambda item: item[1])
        for dof_id, idx in ordered_dofs:
            print(f"K[{idx}, :] -> global DOF {dof_id}")

        return K

    def print_reduced_stiffness_with_dofs(self):
        self.initialize_active_dofs()
        K = self.assemble_global_stiffness()
        K_red, reduced_dofs = self._condense_dynamic_stiffness(K)

        print("Reduced stiffness matrix K_red (dynamic DOFs):")
        print(K_red)
        print("\nDOF mapping for K_red rows/columns:")

        for idx, dof_id in enumerate(reduced_dofs):
            print(f"K_red[{idx}, :] -> global DOF {dof_id}")

        return K_red

    def print_global_geometric_stiffness_with_dofs(self):
        """
        Vytiskne globální geometrickou matici tuhosti (počátečních napětí) K_sigma
        a odpovídající mapování řádků/sloupců na globální DOF.
        """
        self._validate_stability_qx()
        self.solve()

        axial_forces = self.get_stability_axial_forces()
        K_sigma = self.assemble_global_geometric_stiffness(axial_forces)

        print("Global geometric stiffness matrix K_sigma:")
        print(K_sigma)
        print("\nDOF mapping for K_sigma rows/columns:")

        ordered_dofs = sorted(self.dof_map.items(), key=lambda item: item[1])
        for dof_id, idx in ordered_dofs:
            print(f"K_sigma[{idx}, :] -> global DOF {dof_id}")

        print("\nElement axial forces used for K_sigma:")
        for elem_id in sorted(axial_forces):
            print(f"N[{elem_id}] = {axial_forces[elem_id]:.6e}")

        return K_sigma


    #Sestavení plného vektoru U, včetně těch fixovaných DOF pro výpočet reakcí
    def build_full_U(self):

        max_dof = max(self.dof_map.keys())

        U_full = np.zeros(max_dof)

        for dof, idx in self.dof_map.items():
            U_full[dof - 1] = self.U[idx]

        return U_full
        
        
    #výpočet reakcí v zafiksovaných DOF
    def compute_reactions(self):

        K_full = self.assemble_global_stiffness_full()
        F_full = self.assemble_global_load_vector_full()

        U_full = self.build_full_U()

        R = K_full @ U_full - F_full

        return R

    #Výpočet koncových sil z posunů na všech prvcích
    def compute_element_forces(self, U):
        forces = {}
        for e_id, e in self.elements.items():
            sec = self.sections[e.section_id]
            L, alpha = self.element_geometry(e_id)
            ids = self.element_dof_ids(e_id)
            u_global = U[ids]
            u_local = local_displacements(u_global, alpha)
            # najít zatížení prvku
            qx, qz = 0.0, 0.0
            for load in self.element_loads:
                if load.element == e_id:
                    qx, qz = load.qx, load.qz
            f_local = element_local_forces(
                sec.E, sec.A, sec.I, L, u_local, qx, qz
            )
            forces[e_id] = f_local
        return forces

    #Výpočet koncových reakcí
    def compute_reactions(self):

        R = {}

        for sup in self.supports:

            dn = self.dof_nodes[sup["node"]]

            for dof in [dn.ux, dn.uy, dn.rz]:

                if dof in self.dof_map:
                    continue   # není fixní

                # reakce = vnitřní síla v DOF
                R[dof] = self.compute_internal_force(dof)

        return R

    #funkce výše používá pomocnou funkci
    def compute_internal_force(self, dof):

        force = 0.0

        for elem in self.elements.values():

            dof_ids = self.element_dof_ids(elem.id)

            if dof not in dof_ids:
                continue

            Kg = self.element_global_stiffness(elem)

            u_elem = self.get_element_displacements(elem)

            idx = dof_ids.index(dof)

            #Vnitřní síla od tuhosti pruru K * U
            force += Kg[idx, :] @ u_elem

            # odečíst zatížení prutu správně mapované
            for eload in self.element_loads:
                if eload.element != elem.id:
                    continue

                L, alpha = self.element_geometry(elem.id)
                Fe = global_element_load(eload.qx, eload.qz, L, alpha)

                for i, d in enumerate(dof_ids):
                    if d == dof:
                        force -= Fe[i]

        return force

    #posuny na prvku
    def get_element_displacements(self, elem):

        dof_ids = self.element_dof_ids(elem.id)

        u_elem = np.zeros(6)

        for i, dof in enumerate(dof_ids):
            u_elem[i] = self.get_dof_value(dof)

        return u_elem


    #Pomocna funkce
    def get_dof_value(self, dof):

        if dof == 0:
            return 0.0

        idx = self.dof_map.get(dof)

        if idx is None:
            return 0.0   # fixní nebo neaktivní DOF

        return self.U[idx]
    
    #vykreslení deformovaného tvaru kce
    def plot_deformed_shape(self, scale=None, n_points=20, show=False):
        # auto scale
        if scale is None:
            max_disp = max(abs(self.U))
            if max_disp == 0:
                scale = 1
            else:
                size = max(
                    max(n.x for n in self.nodes.values()) -
                    min(n.x for n in self.nodes.values()),
                    max(n.y for n in self.nodes.values()) -
                    min(n.y for n in self.nodes.values())
                )
                scale = 0.1 * size / max_disp

        fig, ax = plt.subplots()
        plotted_x = []
        plotted_y = []

        # původní konstrukce
        for elem in self.elements.values():
            di = self.dof_nodes[elem.i]
            dj = self.dof_nodes[elem.j]

            ni = self.nodes[di.node_id]
            nj = self.nodes[dj.node_id]

            xi, yi = ni.x, ni.y
            xj, yj = nj.x, nj.y
            ax.plot([xi, xj], [yi, yj], 'k--', linewidth=1)

        #Vykresli podpory
        self.plot_releases(ax) #vnitřní klouby    
        self.plot_supports(ax) #vnější vazby

        # deformovaný tvar prutu
        for elem in self.elements.values():

            i, j = elem.i, elem.j

            # geometrie
            L, alpha = self.element_geometry(elem.id)
            c = np.cos(alpha)
            s = np.sin(alpha)

            di = self.dof_nodes[elem.i]
            dj = self.dof_nodes[elem.j]

            ni = self.nodes[di.node_id]
            nj = self.nodes[dj.node_id]

            xi, yi = ni.x, ni.y
            xj, yj = nj.x, nj.y

            # globální DOF indexy
            di = self.dof_nodes[i]
            dj = self.dof_nodes[j]

            # globální vektor prvku
            u_global = np.array([
                self.get_dof_value(di.ux),
                self.get_dof_value(di.uy),
                self.get_dof_value(di.rz),
                self.get_dof_value(dj.ux),
                self.get_dof_value(dj.uy),
                self.get_dof_value(dj.rz),
            ])

            # transformační matice
            T = np.array([
                [ c,  s, 0,  0,  0, 0],
                [-s,  c, 0,  0,  0, 0],
                [ 0,  0, 1,  0,  0, 0],
                [ 0,  0, 0,  c,  s, 0],
                [ 0,  0, 0, -s,  c, 0],
                [ 0,  0, 0,  0,  0, 1],
            ])

            u_local = T @ u_global

            v1 = u_local[1]
            phi1 = u_local[2]
            v2 = u_local[4]
            phi2 = u_local[5]
            u1 = u_local[0]
            u2 = u_local[3]

            # vykreslení křivky
            xs = []
            ys = []

            vmax = 0
            vmax_pos = None

            for xi_loc in np.linspace(0, 1, n_points):

                N1 = 1 - 3*xi_loc**2 + 2*xi_loc**3
                N2 = L*(xi_loc - 2*xi_loc**2 + xi_loc**3)
                N3 = 3*xi_loc**2 - 2*xi_loc**3
                N4 = L*(-xi_loc**2 + xi_loc**3)

                v = N1*v1 + N2*phi1 + N3*v2 + N4*phi2
                

                x_local = xi_loc * L
                # axiální interpolace
                u_axial = (1 - xi_loc) * u1 + xi_loc * u2

                # zpět do globálu
                xg = xi + c*(x_local + scale*u_axial) - s*(scale*v)
                yg = yi + s*(x_local + scale*u_axial) + c*(scale*v)

                xs.append(xg)
                ys.append(yg)

                if abs(v) > vmax:
                    vmax = abs(v)
                    vmax_pos = (xg, yg)

            ax.plot(xs, ys, 'r', linewidth=2)
            plotted_x.extend(xs)
            plotted_y.extend(ys)
            #Popisek maxima na prutu
            if vmax_pos:
                ax.text(
                    vmax_pos[0],
                    vmax_pos[1],
                    f"{vmax:.3e}",
                    fontsize=8,
                    color="green",
                    bbox=dict(facecolor="white", alpha=0.7)
                )

            
        # --- popisky uzlových deformací + nalezení maxima ---

        max_val = 0.0
        max_pos = None

        for node_id, node in self.nodes.items():

            d = self.dof_nodes[node_id]

            ux = self.get_dof_value(d.ux)
            uy = self.get_dof_value(d.uy)

            disp = np.sqrt(ux**2 + uy**2)

            # deformovaná poloha uzlu
            x_def = node.x + scale * ux
            y_def = node.y + scale * uy
            plotted_x.append(x_def)
            plotted_y.append(y_def)

            # popisek v uzlu
            ax.text(
                x_def,
                y_def,
                f"{disp:.3e}",
                fontsize=8,
                color="blue",
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )

        #další nastavení grafu    
        self._apply_plot_bounds(ax, plotted_x, plotted_y)
        plt.title(r"Deformation and values $(u^2+v^2)^{1/2}$")
        if show:
            plt.show()

    def plot_mode_shape(self, mode_index, scale=None, n_points=20, show=False):
        U_mode = self._mode_shape_full_vector(mode_index)

        if scale is None:
            max_disp = np.max(np.abs(U_mode))
            if max_disp == 0:
                scale = 1.0
            else:
                size = max(
                    max(n.x for n in self.nodes.values()) - min(n.x for n in self.nodes.values()),
                    max(n.y for n in self.nodes.values()) - min(n.y for n in self.nodes.values()),
                )
                scale = 0.1 * size / max_disp

        fig, ax = plt.subplots()
        plotted_x = []
        plotted_y = []

        for elem in self.elements.values():
            di = self.dof_nodes[elem.i]
            dj = self.dof_nodes[elem.j]
            ni = self.nodes[di.node_id]
            nj = self.nodes[dj.node_id]
            ax.plot([ni.x, nj.x], [ni.y, nj.y], 'k--', linewidth=1)

        self.plot_releases(ax)
        self.plot_supports(ax)

        for elem in self.elements.values():
            i, j = elem.i, elem.j
            L, alpha = self.element_geometry(elem.id)
            c = np.cos(alpha)
            s = np.sin(alpha)

            di = self.dof_nodes[i]
            dj = self.dof_nodes[j]
            ni = self.nodes[di.node_id]
            nj = self.nodes[dj.node_id]
            xi, yi = ni.x, ni.y

            u_global = np.array([
                U_mode[di.ux - 1],
                U_mode[di.uy - 1],
                U_mode[di.rz - 1],
                U_mode[dj.ux - 1],
                U_mode[dj.uy - 1],
                U_mode[dj.rz - 1],
            ])

            T = np.array([
                [c, s, 0, 0, 0, 0],
                [-s, c, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [0, 0, 0, 0, 0, 1],
            ])
            u_local = T @ u_global

            v1, phi1, v2, phi2 = u_local[1], u_local[2], u_local[4], u_local[5]
            u1, u2 = u_local[0], u_local[3]

            xs = []
            ys = []
            for xi_loc in np.linspace(0, 1, n_points):
                N1 = 1 - 3 * xi_loc ** 2 + 2 * xi_loc ** 3
                N2 = L * (xi_loc - 2 * xi_loc ** 2 + xi_loc ** 3)
                N3 = 3 * xi_loc ** 2 - 2 * xi_loc ** 3
                N4 = L * (-xi_loc ** 2 + xi_loc ** 3)

                v = N1 * v1 + N2 * phi1 + N3 * v2 + N4 * phi2
                x_local = xi_loc * L
                u_axial = (1 - xi_loc) * u1 + xi_loc * u2

                xg = xi + c * (x_local + scale * u_axial) - s * (scale * v)
                yg = yi + s * (x_local + scale * u_axial) + c * (scale * v)
                xs.append(xg)
                ys.append(yg)

            ax.plot(xs, ys, 'r', linewidth=2)
            plotted_x.extend(xs)
            plotted_y.extend(ys)

        omega = np.sqrt(max(self.eigenvalues[mode_index], 0.0))
        freq_hz = omega / (2 * np.pi)
        ax.set_title(fr"Shape mode no. {mode_index + 1}: $f$ = {freq_hz:.3f} Hz")
        self._apply_plot_bounds(ax, plotted_x, plotted_y)
        ax.set_axis_off()

        if show:
            plt.show()

    def plot_all_mode_shapes(self, show=False):
        n_modes = min(self.number_of_eigenvectors, self.eigenvectors.shape[1] if self.eigenvectors.ndim == 2 else 0)
        for i in range(n_modes):
            self.plot_mode_shape(i, show=False)
        if show:
            self.show_all_plots()

    def clear_animation_references(self):
        self._last_animation = None
        self._animations = []

    def _infer_animation_writer(self, output_path, writer=None):
        if writer is not None:
            return writer

        suffix = output_path.lower().rsplit(".", 1)
        extension = suffix[1] if len(suffix) == 2 else ""
        writer_by_extension = {
            "gif": "pillow",
            "mp4": "ffmpeg",
            "m4v": "ffmpeg",
            "mov": "ffmpeg",
            "html": "html",
            "htm": "html",
        }
        if extension not in writer_by_extension:
            raise ValueError(
                "Unsupported animation export format. Use .gif, .mp4, .mov, .m4v, or .html, "
                "or pass writer='pillow'/'ffmpeg'/'html'."
            )
        return writer_by_extension[extension]

    def _build_animation_writer(self, output_path, writer=None, fps=None):
        selected_writer = self._infer_animation_writer(output_path, writer=writer)
        if fps is None:
            fps = 20
        fps = max(float(fps), 1.0)

        if selected_writer == "pillow":
            return PillowWriter(fps=fps)
        if selected_writer == "ffmpeg":
            if not FFMpegWriter.isAvailable():
                raise RuntimeError(
                    "FFmpeg writer is not available in the current environment. "
                    "Use GIF export (.gif) or install FFmpeg for MP4/MOV export."
                )
            return FFMpegWriter(fps=fps)
        if selected_writer == "html":
            return HTMLWriter(fps=fps)
        raise ValueError(f"Unsupported animation writer '{selected_writer}'.")

    def export_animation(self, output_path, animation=None, writer=None, fps=None, dpi=120):
        if animation is None:
            animation = self._last_animation
        if animation is None:
            raise ValueError("No animation is available for export. Create an animation first.")

        animation_writer = self._build_animation_writer(output_path, writer=writer, fps=fps)
        animation.save(output_path, writer=animation_writer, dpi=dpi)
        return output_path

    def export_last_animation(self, output_path, writer=None, fps=None, dpi=120):
        return self.export_animation(output_path, animation=self._last_animation, writer=writer, fps=fps, dpi=dpi)

    def _create_animation_title(self, fig, initial_text=""):
        fig.subplots_adjust(top=0.84)
        return fig.text(
            0.5,
            0.965,
            initial_text,
            ha="center",
            va="top",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.55,
                "pad": 4,
            },
        )

    def _structure_bounds(self):
        xs = [node.x for node in self.nodes.values()]
        ys = [node.y for node in self.nodes.values()]
        if not xs or not ys:
            return 0.0, 1.0, 0.0, 1.0
        return min(xs), max(xs), min(ys), max(ys)

    def _apply_plot_bounds(self, ax, x_values=None, y_values=None, margin_ratio=0.12, min_margin_ratio=0.04):
        x_min, x_max, y_min, y_max = self._structure_bounds()

        if x_values:
            x_min = min(x_min, min(x_values))
            x_max = max(x_max, max(x_values))
        if y_values:
            y_min = min(y_min, min(y_values))
            y_max = max(y_max, max(y_values))

        dx = max(x_max - x_min, 1e-9)
        dy = max(y_max - y_min, 1e-9)
        char_size = max(dx, dy, 1.0)
        margin = max(margin_ratio * char_size, min_margin_ratio * char_size)

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_aspect("equal", adjustable="box")

    def _dynamic_animation_setup(self, scale=None, n_points=20, frames_per_period=60):
        harmonic_full_vectors, omega, period, time_values = self._dynamic_time_history_vectors(frames_per_period)
        if scale is None:
            max_disp = max(np.max(np.abs(U_hat)) for U_hat in harmonic_full_vectors.values())
            if max_disp == 0:
                scale = 1.0
            else:
                size = max(
                    max(n.x for n in self.nodes.values()) - min(n.x for n in self.nodes.values()),
                    max(n.y for n in self.nodes.values()) - min(n.y for n in self.nodes.values()),
                )
                scale = 0.1 * size / max_disp

        interpolation_points = np.linspace(0.0, 1.0, max(int(n_points), 2))
        element_data = []
        for elem in self.elements.values():
            di = self.dof_nodes[elem.i]
            dj = self.dof_nodes[elem.j]
            ni = self.nodes[di.node_id]
            nj = self.nodes[dj.node_id]
            L, alpha = self.element_geometry(elem.id)
            c = np.cos(alpha)
            s = np.sin(alpha)
            transform = transformation_matrix(alpha)
            base_x = ni.x + c * (interpolation_points * L)
            base_y = ni.y + s * (interpolation_points * L)
            N1 = 1 - 3 * interpolation_points ** 2 + 2 * interpolation_points ** 3
            N2 = L * (interpolation_points - 2 * interpolation_points ** 2 + interpolation_points ** 3)
            N3 = 3 * interpolation_points ** 2 - 2 * interpolation_points ** 3
            N4 = L * (-interpolation_points ** 2 + interpolation_points ** 3)
            axial_i = 1 - interpolation_points
            axial_j = interpolation_points
            element_data.append({
                "element": elem,
                "di": di,
                "dj": dj,
                "node_i": ni,
                "node_j": nj,
                "length": L,
                "cos": c,
                "sin": s,
                "transform": transform,
                "base_x": base_x,
                "base_y": base_y,
                "N1": N1,
                "N2": N2,
                "N3": N3,
                "N4": N4,
                "axial_i": axial_i,
                "axial_j": axial_j,
            })

        sampled_time_values = self._sample_time_values(time_values)
        bounds_x = []
        bounds_y = []
        for t in sampled_time_values:
            U_t = self._compose_dynamic_frame_vector(harmonic_full_vectors, omega, t)
            for data in element_data:
                xs, ys = self._dynamic_deformed_coordinates(data, U_t, scale)
                bounds_x.extend(xs)
                bounds_y.extend(ys)

        return {
            "harmonic_full_vectors": harmonic_full_vectors,
            "omega": omega,
            "period": period,
            "time_values": time_values,
            "scale": scale,
            "element_data": element_data,
            "bounds_x": bounds_x,
            "bounds_y": bounds_y,
        }

    def _dynamic_deformed_coordinates(self, element_data, displacement_vector, scale):
        di = element_data["di"]
        dj = element_data["dj"]
        u_global = np.array([
            displacement_vector[di.ux - 1],
            displacement_vector[di.uy - 1],
            displacement_vector[di.rz - 1],
            displacement_vector[dj.ux - 1],
            displacement_vector[dj.uy - 1],
            displacement_vector[dj.rz - 1],
        ], dtype=float)
        u_local = element_data["transform"] @ u_global
        u1, v1, phi1, u2, v2, phi2 = u_local
        v = (
            element_data["N1"] * v1
            + element_data["N2"] * phi1
            + element_data["N3"] * v2
            + element_data["N4"] * phi2
        )
        u_axial = element_data["axial_i"] * u1 + element_data["axial_j"] * u2
        xs = element_data["base_x"] + scale * (element_data["cos"] * u_axial - element_data["sin"] * v)
        ys = element_data["base_y"] + scale * (element_data["sin"] * u_axial + element_data["cos"] * v)
        return xs, ys

    def plot_dynamic_deformation_animation(self, scale=None, n_points=20, frames_per_period=60, show=False):
        return self.plot_dynamic_response_animation(
            scale=scale,
            n_points=n_points,
            frames_per_period=frames_per_period,
            show=show,
        )

    def plot_dynamic_response_animation(self, scale=None, n_points=20, frames_per_period=60, show=False):
        animation_data = self._dynamic_animation_setup(scale=scale, n_points=n_points, frames_per_period=frames_per_period)
        harmonic_full_vectors = animation_data["harmonic_full_vectors"]
        omega = animation_data["omega"]
        period = animation_data["period"]
        time_values = animation_data["time_values"]
        scale = animation_data["scale"]
        element_data = animation_data["element_data"]

        fig, ax = plt.subplots()
        title_artist = self._create_animation_title(fig)

        for data in element_data:
            ni = data["node_i"]
            nj = data["node_j"]
            ax.plot([ni.x, nj.x], [ni.y, nj.y], "k--", linewidth=1)

        self.plot_releases(ax)
        self.plot_supports(ax)

        deformed_lines = []
        for _ in element_data:
            line, = ax.plot([], [], "r", linewidth=2)
            deformed_lines.append(line)

        ax.axis("equal")
        ax.set_axis_off()

        def update(frame_index):
            t = time_values[frame_index]
            U_t = self._compose_dynamic_frame_vector(harmonic_full_vectors, omega, t)
            for line, data in zip(deformed_lines, element_data):
                xs, ys = self._dynamic_deformed_coordinates(data, U_t, scale)
                line.set_data(xs, ys)
            title_artist.set_text(
                fr"Harmonic response: $u(t)=\Re\left(\sum_k \hat{{u}}_k e^{{ik\Omega t}}\right)$, "
                fr"$t={t:.3e}\,\mathrm{{s}}$, $T={period:.3e}\,\mathrm{{s}}$"
            )
            return deformed_lines

        self._apply_plot_bounds(ax, animation_data["bounds_x"], animation_data["bounds_y"])
        interval_ms = 1000 * period / len(time_values)
        self._last_animation = FuncAnimation(
            fig,
            update,
            frames=len(time_values),
            interval=interval_ms,
            blit=False,
            repeat=True,
        )
        self._animations.append(self._last_animation)

        update(0)
        if show:
            plt.show()

    def _dynamic_time_history_vectors(self, frames_per_period):
        if self.dynamic_excitation_frequency is None or self.dynamic_response.size == 0:
            raise ValueError("Dynamic steady-state solution must be solved before animation.")
        if self.dynamic_excitation_frequency <= 0:
            raise ValueError("Omega must be positive to animate one period.")

        harmonic_full_vectors = {
            multiplier: self._dynamic_response_full_vector(response)
            for multiplier, response in self.dynamic_harmonic_responses.items()
        }
        if not harmonic_full_vectors:
            harmonic_full_vectors = {1: self._dynamic_response_full_vector()}

        omega = float(self.dynamic_excitation_frequency)
        period = 2 * np.pi / omega
        time_values = np.linspace(0.0, period, max(int(frames_per_period), 2), endpoint=False)
        return harmonic_full_vectors, omega, period, time_values

    def _compose_dynamic_frame_vector(self, harmonic_full_vectors, omega, t):
        U_t = np.zeros_like(next(iter(harmonic_full_vectors.values())), dtype=float)
        for multiplier, U_hat in harmonic_full_vectors.items():
            U_t += np.real(U_hat * np.exp(1j * multiplier * omega * t))
        return U_t

    def dynamic_displacement_vector_at_time(self, t, reduced=False):
        if self.dynamic_excitation_frequency is None or self.dynamic_response.size == 0:
            raise ValueError("Dynamic steady-state solution must be solved before evaluating response in time.")

        omega = float(self.dynamic_excitation_frequency)
        if reduced:
            harmonic_vectors = self.dynamic_harmonic_responses or {1: self.dynamic_response}
            U_t = np.zeros_like(next(iter(harmonic_vectors.values())), dtype=float)
            for multiplier, U_hat in harmonic_vectors.items():
                U_t += np.real(U_hat * np.exp(1j * multiplier * omega * t))
            return U_t

        harmonic_full_vectors = {
            multiplier: self._dynamic_response_full_vector(response)
            for multiplier, response in (self.dynamic_harmonic_responses or {1: self.dynamic_response}).items()
        }
        return self._compose_dynamic_frame_vector(harmonic_full_vectors, omega, t)

    def dynamic_deformation_in_dof(self, dof_id, t):
        displacement_vector = self.dynamic_displacement_vector_at_time(t)
        if dof_id < 1 or dof_id > displacement_vector.size:
            return 0.0
        return float(np.real(displacement_vector[dof_id - 1]))

    def dynamic_element_end_forces(self, elem, t):
        displacement_vector = self.dynamic_displacement_vector_at_time(t)
        return self.element_end_forces(elem, displacement_vector=displacement_vector)

    def dynamic_reaction_in_dof(self, dof_id, t):
        if dof_id in self.dof_map:
            return None

        force = 0.0
        displacement_vector = self.dynamic_displacement_vector_at_time(t)

        for elem in self.elements.values():
            dof_ids = self.element_dof_ids(elem.id)
            if dof_id not in dof_ids:
                continue

            Kg = self.element_global_stiffness(elem)
            u_elem = np.array([displacement_vector[d - 1] if d > 0 and d <= displacement_vector.size else 0.0 for d in dof_ids], dtype=float)
            idx = dof_ids.index(dof_id)
            force += Kg[idx, :] @ u_elem

            for eload in self.element_loads:
                if eload.element != elem.id:
                    continue
                L, alpha = self.element_geometry(elem.id)
                Fe = global_element_load(eload.qx, eload.qz, L, alpha)
                force -= Fe[idx]

        return float(np.real(force))

    def _sample_time_values(self, time_values, max_samples=12):
        if len(time_values) <= max_samples:
            return time_values
        sample_indices = np.linspace(0, len(time_values) - 1, max_samples, dtype=int)
        return time_values[sample_indices]

    def _internal_force_diagram_data(self, elem, kind, scale, displacement_vector, npts=40):
        di = self.dof_nodes[elem.i]
        dj = self.dof_nodes[elem.j]

        ni = self.nodes[di.node_id]
        nj = self.nodes[dj.node_id]

        xi, yi = ni.x, ni.y
        xj, yj = nj.x, nj.y
        dx = xj - xi
        dy = yj - yi
        L = np.hypot(dx, dy)
        cx = dx / L
        cy = dy / L
        nx = -cy
        ny = cx
        if kind == "M":
            nx = -nx
            ny = -ny

        forces = self.element_end_forces(elem, displacement_vector=displacement_vector)
        qz = self.get_element_qz(elem)
        xs = np.linspace(0, L, npts)

        X = []
        Y = []
        Xbase = []
        Ybase = []
        values = []
        for x in xs:
            N, V, M = element_diagram(x, L, forces, qz)
            if kind == "N":
                val = np.real_if_close(N)
            elif kind == "V":
                val = np.real_if_close(V)
            else:
                val = np.real_if_close(M)
            val = float(np.real(val))

            xb = xi + cx * x
            yb = yi + cy * x
            xd = xb + nx * val * scale
            yd = yb + ny * val * scale
            X.append(xd)
            Y.append(yd)
            Xbase.append(xb)
            Ybase.append(yb)
            values.append(val)

        return Xbase, Ybase, X, Y, values

    def plot_dynamic_normal_force_animation(self, scale=None, frames_per_period=60, npts=40, show=False):
        return self.plot_dynamic_internal_force_animation(kind="N", scale=scale, frames_per_period=frames_per_period, npts=npts, show=show)

    def plot_dynamic_shear_force_animation(self, scale=None, frames_per_period=60, npts=40, show=False):
        return self.plot_dynamic_internal_force_animation(kind="V", scale=scale, frames_per_period=frames_per_period, npts=npts, show=show)

    def plot_dynamic_bending_moment_animation(self, scale=None, frames_per_period=60, npts=40, show=False):
        return self.plot_dynamic_internal_force_animation(kind="M", scale=scale, frames_per_period=frames_per_period, npts=npts, show=show)

    def plot_dynamic_internal_force_animation(self, kind="M", scale=None, frames_per_period=60, npts=40, show=False):
        kind = kind.upper()
        if kind not in ("N", "V", "M"):
            raise ValueError("kind must be one of 'N', 'V', 'M'")

        harmonic_full_vectors, omega, period, time_values = self._dynamic_time_history_vectors(frames_per_period)

        if scale is None:
            max_val = 0.0
            sampled_time_values = self._sample_time_values(time_values)
            for t in sampled_time_values:
                U_t = self._compose_dynamic_frame_vector(harmonic_full_vectors, omega, t)
                for elem in self.elements.values():
                    forces = self.element_end_forces(elem, displacement_vector=U_t)
                    if kind == "N":
                        values = [forces[0], forces[3]]
                    elif kind == "V":
                        values = [forces[1], forces[4]]
                    else:
                        values = [forces[2], forces[5]]
                    max_val = max(max_val, max(abs(float(np.real(v))) for v in values))
            if max_val == 0:
                scale = 1.0
            else:
                total_length = sum(self.element_geometry(elem.id)[0] for elem in self.elements.values())
                L_ref = total_length / len(self.elements)
                scale = 0.2 * L_ref / max_val

        fig, ax = plt.subplots()
        title_artist = self._create_animation_title(fig)

        diagram_lines = []
        fill_patches = []
        for elem in self.elements.values():
            di = self.dof_nodes[elem.i]
            dj = self.dof_nodes[elem.j]
            ni = self.nodes[di.node_id]
            nj = self.nodes[dj.node_id]
            ax.plot([ni.x, nj.x], [ni.y, nj.y], "k-", lw=1)

            line, = ax.plot([], [], "r", linewidth=2)
            patch = patches.Polygon([[ni.x, ni.y]], closed=True, color="r", alpha=0.25)
            ax.add_patch(patch)
            diagram_lines.append(line)
            fill_patches.append(patch)

        self.plot_releases(ax)
        self.plot_supports(ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()

        def update(frame_index):
            t = time_values[frame_index]
            U_t = self._compose_dynamic_frame_vector(harmonic_full_vectors, omega, t)

            for line, patch, elem in zip(diagram_lines, fill_patches, self.elements.values()):
                Xbase, Ybase, X, Y, _ = self._internal_force_diagram_data(elem, kind, scale, U_t, npts=npts)
                line.set_data(X, Y)
                polygon_points = np.column_stack([
                    np.array(X + list(reversed(Xbase))),
                    np.array(Y + list(reversed(Ybase)))
                ])
                patch.set_xy(polygon_points)

            title_artist.set_text(
                fr"Dynamic {kind} diagram: $t={t:.3e}\,\mathrm{{s}}$, $T={period:.3e}\,\mathrm{{s}}$"
            )
            return diagram_lines + fill_patches

        bounds_x = []
        bounds_y = []
        sampled_time_values = self._sample_time_values(time_values)
        for t in sampled_time_values:
            U_t = self._compose_dynamic_frame_vector(harmonic_full_vectors, omega, t)
            for elem in self.elements.values():
                Xbase, Ybase, X, Y, _ = self._internal_force_diagram_data(elem, kind, scale, U_t, npts=npts)
                bounds_x.extend(Xbase)
                bounds_x.extend(X)
                bounds_y.extend(Ybase)
                bounds_y.extend(Y)
        self._apply_plot_bounds(ax, bounds_x, bounds_y)
        interval_ms = 1000 * period / len(time_values)
        self._last_animation = FuncAnimation(
            fig,
            update,
            frames=len(time_values),
            interval=interval_ms,
            blit=False,
            repeat=True,
        )
        self._animations.append(self._last_animation)

        update(0)
        if show:
            plt.show()

    def plot_dynamic_results_dashboard(self, displacement_scale=None, force_scales=None, n_points=20, frames_per_period=60, npts=40, show=False):
        animation_data = self._dynamic_animation_setup(
            scale=displacement_scale,
            n_points=n_points,
            frames_per_period=frames_per_period,
        )
        harmonic_full_vectors = animation_data["harmonic_full_vectors"]
        omega = animation_data["omega"]
        period = animation_data["period"]
        time_values = animation_data["time_values"]
        element_data = animation_data["element_data"]
        displacement_scale = animation_data["scale"]

        if force_scales is None:
            force_scales = {}
        computed_force_scales = {}
        sampled_time_values = self._sample_time_values(time_values)
        for kind in ("M", "V", "N"):
            explicit_scale = force_scales.get(kind)
            if explicit_scale is not None:
                computed_force_scales[kind] = explicit_scale
                continue
            max_val = 0.0
            for t in sampled_time_values:
                U_t = self._compose_dynamic_frame_vector(harmonic_full_vectors, omega, t)
                for elem in self.elements.values():
                    forces = self.element_end_forces(elem, displacement_vector=U_t)
                    if kind == "N":
                        values = [forces[0], forces[3]]
                    elif kind == "V":
                        values = [forces[1], forces[4]]
                    else:
                        values = [forces[2], forces[5]]
                    max_val = max(max_val, max(abs(float(np.real(v))) for v in values))
            if max_val == 0:
                computed_force_scales[kind] = 1.0
            else:
                total_length = sum(self.element_geometry(elem.id)[0] for elem in self.elements.values())
                L_ref = total_length / len(self.elements)
                computed_force_scales[kind] = 0.2 * L_ref / max_val

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.subplots_adjust(top=0.90, wspace=0.08, hspace=0.08)
        title_artist = self._create_animation_title(fig)
        ax_response = axes[0, 0]
        diagram_axes = {
            "M": axes[0, 1],
            "V": axes[1, 0],
            "N": axes[1, 1],
        }

        for data in element_data:
            ni = data["node_i"]
            nj = data["node_j"]
            ax_response.plot([ni.x, nj.x], [ni.y, nj.y], "k--", linewidth=1)
        self.plot_releases(ax_response)
        self.plot_supports(ax_response)
        response_lines = []
        for _ in element_data:
            line, = ax_response.plot([], [], "r", linewidth=2)
            response_lines.append(line)
        self._apply_plot_bounds(ax_response, animation_data["bounds_x"], animation_data["bounds_y"])
        ax_response.set_axis_off()
        ax_response.set_title("Deformed shape")

        diagram_artists = {}
        for kind, ax in diagram_axes.items():
            lines = []
            patches_list = []
            bounds_x = []
            bounds_y = []
            for elem in self.elements.values():
                di = self.dof_nodes[elem.i]
                dj = self.dof_nodes[elem.j]
                ni = self.nodes[di.node_id]
                nj = self.nodes[dj.node_id]
                ax.plot([ni.x, nj.x], [ni.y, nj.y], "k-", lw=1)
                line, = ax.plot([], [], "r", linewidth=2)
                patch = patches.Polygon([[ni.x, ni.y]], closed=True, color="r", alpha=0.25)
                ax.add_patch(patch)
                lines.append(line)
                patches_list.append(patch)
            self.plot_releases(ax)
            self.plot_supports(ax)
            for t in sampled_time_values:
                U_t = self._compose_dynamic_frame_vector(harmonic_full_vectors, omega, t)
                for elem in self.elements.values():
                    Xbase, Ybase, X, Y, _ = self._internal_force_diagram_data(elem, kind, computed_force_scales[kind], U_t, npts=npts)
                    bounds_x.extend(Xbase)
                    bounds_x.extend(X)
                    bounds_y.extend(Ybase)
                    bounds_y.extend(Y)
            self._apply_plot_bounds(ax, bounds_x, bounds_y)
            ax.set_axis_off()
            ax.set_title(f"{kind} diagram")
            diagram_artists[kind] = {"lines": lines, "patches": patches_list}

        def update(frame_index):
            t = time_values[frame_index]
            U_t = self._compose_dynamic_frame_vector(harmonic_full_vectors, omega, t)
            for line, data in zip(response_lines, element_data):
                xs, ys = self._dynamic_deformed_coordinates(data, U_t, displacement_scale)
                line.set_data(xs, ys)

            updated_artists = list(response_lines)
            for kind, artists in diagram_artists.items():
                for line, patch, elem in zip(artists["lines"], artists["patches"], self.elements.values()):
                    Xbase, Ybase, X, Y, _ = self._internal_force_diagram_data(elem, kind, computed_force_scales[kind], U_t, npts=npts)
                    line.set_data(X, Y)
                    polygon_points = np.column_stack([
                        np.array(X + list(reversed(Xbase))),
                        np.array(Y + list(reversed(Ybase))),
                    ])
                    patch.set_xy(polygon_points)
                updated_artists.extend(artists["lines"])
                updated_artists.extend(artists["patches"])

            title_artist.set_text(
                fr"Dynamic steady-state response: $t={t:.3e}\,\mathrm{{s}}$, $T={period:.3e}\,\mathrm{{s}}$"
            )
            return updated_artists

        interval_ms = 1000 * period / len(time_values)
        self._last_animation = FuncAnimation(
            fig,
            update,
            frames=len(time_values),
            interval=interval_ms,
            blit=False,
            repeat=True,
        )
        self._animations.append(self._last_animation)

        update(0)
        if show:
            plt.show()
    #print reakce
    def format_reactions(self):
        R = self.compute_reactions()

        lines = ["REACTIONS:"]

        for dn in self.dof_nodes.values():
            node = dn.node_id

            rx = R.get(dn.ux)
            ry = R.get(dn.uy)
            mz = R.get(dn.rz)

            if rx is not None:
                lines.append(f"Node {node} Rx = {rx:.6e}")

            if ry is not None:
                lines.append(f"Node {node} Ry = {ry:.6e}")

            if mz is not None:
                lines.append(f"Node {node} Mz = {mz:.6e}")

        if len(lines) == 1:
            lines.append("Žádné reakce k zobrazení.")

        return "\n".join(lines)

    def print_reactions(self):
        print("\n" + self.format_reactions())

    def reaction_in_dof(self, dof_id):
        return self.compute_reactions().get(dof_id)

    def _displacement_value_from_vector(self, displacement_vector, dof_id):
        if dof_id == 0:
            return 0.0

        displacement_vector = np.asarray(displacement_vector)

        if displacement_vector.ndim != 1:
            raise ValueError("displacement_vector must be a 1D array-like sequence.")

        vector_size = displacement_vector.shape[0]
        max_dof = max(max(dn.ux, dn.uy, dn.rz) for dn in self.dof_nodes.values())

        if vector_size >= max_dof:
            return displacement_vector[dof_id - 1]

        idx = self.dof_map.get(dof_id)
        if idx is None:
            return 0.0
        if idx >= vector_size:
            raise IndexError(
                f"Displacement vector with size {vector_size} does not contain active DOF {dof_id} "
                f"(mapped index {idx})."
            )
        return displacement_vector[idx]

                
    #koncové síly na prvku
    def element_end_forces(self, elem, displacement_vector=None):
        sec = self.sections[elem.section_id]
        L, alpha = self.element_geometry(elem.id)
        # lokální tuhost — VOLÁ SE FUNKCE MIMO TŘÍDU
        k_local = local_stiffness(sec.E, sec.A, sec.I, L)
        # transformace
        T = transformation_matrix(alpha)
        # globální posuny prvku
        if displacement_vector is None:
            u_global = self.get_element_displacements(elem)
        else:
            di = self.dof_nodes[elem.i]
            dj = self.dof_nodes[elem.j]
            u_global = np.array([
                self._displacement_value_from_vector(displacement_vector, di.ux),
                self._displacement_value_from_vector(displacement_vector, di.uy),
                self._displacement_value_from_vector(displacement_vector, di.rz),
                self._displacement_value_from_vector(displacement_vector, dj.ux),
                self._displacement_value_from_vector(displacement_vector, dj.uy),
                self._displacement_value_from_vector(displacement_vector, dj.rz),
            ], dtype=np.asarray(displacement_vector).dtype)
        # převod do lokálního systému
        u_local = T @ u_global
        # koncové síly v lokálním systému
        f_local = k_local @ u_local
        # přičti vliv zatížení prutu
        qx, qz = self.get_element_loads(elem)

        f_load = local_element_load(qx, qz, L)

        # KLÍČOVÝ krok
        f_local = f_local - f_load
        return f_local

    #vykreslení diagramů
    def plot_element_diagram(self, elem, kind="M", scale=1.0, npts=40):
        di = self.dof_nodes[elem.i]
        dj = self.dof_nodes[elem.j]

        ni = self.nodes[di.node_id]
        nj = self.nodes[dj.node_id]
        
        xi, yi = ni.x, ni.y 
        xj, yj = nj.x, nj.y
        dx = xj - xi
        dy = yj - yi
        L = np.hypot(dx, dy)
        cx = dx / L
        cy = dy / L
        # normála
        nx = -cy
        ny = cx
        if kind == "M":
            nx = -nx  #M na stranu tažených vláken
            ny = -ny
            
        # koncové síly
        forces = self.element_end_forces(elem)
        # zatížení
        qz = self.get_element_qz(elem)
        xs = np.linspace(0, L, npts)

        X = []
        Y = []
        Xbase = []
        Ybase = []
        values = []
        for x in xs:
            N, V, M = element_diagram(x, L, forces, qz)
            if kind == "N":
                val = N
            elif kind == "V":
                val = V
            else:
                val = M
                
            # základní bod na prutu    
            xb = xi + cx * x
            yb = yi + cy * x
            #posun diagramu
            xd = xb + nx * val * scale
            yd = yb + ny * val * scale
            X.append(xd)
            Y.append(yd)
            Xbase.append(xb)
            Ybase.append(yb)
            values.append(val)
        # kreslení
        plt.plot(Xbase, Ybase, "k", linewidth=1)
        plt.plot(X, Y, "r", linewidth=2)
        Xp = X + list(reversed(Xbase))
        Yp = Y + list(reversed(Ybase))
        plt.fill(Xp, Yp, color="r", alpha=0.25)
        # kreslení podpor
        ax = plt.gca()
        self.plot_releases(ax) #vnitřní klouby
        self.plot_supports(ax) #vnější vazby

        # popisky grafu
        plotted = []
        # --- levý uzel ---
        self.annotate_value(X[0], Y[0], values[0], plotted)
        # --- pravý uzel ---
        self.annotate_value(X[-1], Y[-1], values[-1], plotted)
        # --- extrém(y) ---
        if kind == "M" and abs(qz) > 1e-12:
            # U M(x) = M1 + V1*x + qz*x^2/2 leží vnitřní extrém v bodě V(x)=0.
            x_ext = -forces[1] / qz
            if 0.0 < x_ext < L:
                _, _, val_ext = element_diagram(x_ext, L, forces, qz)
                xb = xi + cx * x_ext
                yb = yi + cy * x_ext
                xd = xb + nx * val_ext * scale
                yd = yb + ny * val_ext * scale
                self.annotate_value(xd, yd, val_ext, plotted)
        else:
            imax = np.argmax(np.abs(values))
            self.annotate_value(X[imax], Y[imax], values[imax], plotted)

        

        
    def annotate_value(self, x, y, val, plotted, tol=1e-6):
        """ Zapíše popisek jen pokud už tam žádný není. plotted = list již popsaných bodů """
        for xp, yp in plotted:
            if abs(x - xp) < tol and abs(y - yp) < tol:
                return
        plt.text(x, y, f"{val:.2f}", fontsize=9, color="blue")
        plotted.append((x, y))

    def get_element_qz(self, elem):
        for el in self.element_loads:
            if el.element == elem.id:
                return el.qz
        return 0.0

    def get_element_loads(self, elem):

        qx = 0.0
        qz = 0.0

        for el in self.element_loads:
            if el.element == elem.id:
                qx = el.qx
                qz = el.qz
                break

        return qx, qz

    def element_internal_force_extreme(self, element, kind="M", extreme="max"):
        """
        Vrátí minimum nebo maximum vnitřní síly na daném prvku.

        Parameters
        ----------
        element : int | Element
            ID prvku nebo instance Element.
        kind : str
            Typ vnitřní síly: "N", "V" nebo "M".
        extreme : str
            Typ extrému: "min" nebo "max".

        Returns
        -------
        tuple[float, float]
            (x, hodnota) kde x je lokální souřadnice na prvku od i-uzlu.
        """
        if isinstance(element, Element):
            elem = element
        else:
            if element not in self.elements:
                raise KeyError(f"Element {element} does not exist")
            elem = self.elements[element]

        kind = kind.upper()
        if kind not in ("N", "V", "M"):
            raise ValueError("kind must be one of 'N', 'V', 'M'")

        extreme = extreme.lower()
        if extreme not in ("min", "max"):
            raise ValueError("extreme must be either 'min' or 'max'")

        L, _ = self.element_geometry(elem.id)
        forces = self.element_end_forces(elem)
        qz = self.get_element_qz(elem)

        candidates_x = [0.0, L]

        if kind == "M" and abs(qz) > 1e-12:
            x_ext = -forces[1] / qz
            if 0.0 < x_ext < L:
                candidates_x.append(x_ext)

        candidates = []
        for x in candidates_x:
            N, V, M = element_diagram(x, L, forces, qz)
            val = {"N": N, "V": V, "M": M}[kind]
            candidates.append((x, val))

        key_fn = (lambda item: item[1])
        return max(candidates, key=key_fn) if extreme == "max" else min(candidates, key=key_fn)

    def plot_internal_forces(self, kind="M", scale=None, show=False):
        if scale is None:
            scale = self.auto_scale(kind)
        plt.figure()
        plotted_x = []
        plotted_y = []
        # nejdřív konstrukce
        for elem in self.elements.values():
            di = self.dof_nodes[elem.i]
            dj = self.dof_nodes[elem.j]

            ni = self.nodes[di.node_id]
            nj = self.nodes[dj.node_id]
            
            plt.plot([ni.x, nj.x], [ni.y, nj.y], "k-", lw=1)
        # teď vnitřní kluby
        
        # pak diagramy
        for elem in self.elements.values():
            self.plot_element_diagram(elem, kind, scale)
            Xbase, Ybase, X, Y, _ = self._internal_force_diagram_data(elem, kind, scale, self.U)
            plotted_x.extend(Xbase)
            plotted_x.extend(X)
            plotted_y.extend(Ybase)
            plotted_y.extend(Y)
        ax = plt.gca()
        self._apply_plot_bounds(ax, plotted_x, plotted_y)
        plt.title(kind)
        if show:
            plt.show()

    @staticmethod
    def show_all_plots():
        plt.show()

    #autoscale pro graf vnitřních sil
    def auto_scale(self, kind):
        max_val = 0.0
        total_length = 0.0
        for elem in self.elements.values():
            L, _ = self.element_geometry(elem.id)
            total_length += L
            forces = self.element_end_forces(elem)
            if kind == "N":
                values = [forces[0], forces[3]]
            elif kind == "V":
                values = [forces[1], forces[4]]
            elif kind == "M":
                values = [forces[2], forces[5]]
            max_val = max(max_val, max(abs(v) for v in values))
        if max_val == 0:
            return 1.0
        # velikost konstrukce jako referenční měřítko
        L_ref = total_length / len(self.elements)
        scale = 0.2 * L_ref / max_val
        return scale


    #Vykreslení podpor
    def plot_supports(self, ax):

        size = 0.03 * max(
            max(n.x for n in self.nodes.values()) -
            min(n.x for n in self.nodes.values()),
            max(n.y for n in self.nodes.values()) -
            min(n.y for n in self.nodes.values())
        )

        for sup in self.supports:

            node_DOF_id = sup["node"]

            di = self.dof_nodes[node_DOF_id]

            ni = self.nodes[di.node_id]

            x, y = ni.x, ni.y


            fix_ux = sup.get("ux", False)
            fix_uy = sup.get("uy", False)
            fix_rz = sup.get("rz", False)

            # --- vetknutí ---
            if fix_ux and fix_uy and fix_rz:

                rect = patches.Rectangle(
                    (x - size/2, y - size/2),
                    size, size,
                    color="black"
                )
                ax.add_patch(rect)

            # --- kloubová podpora ---
            elif fix_ux and fix_uy:

                triangle = patches.Polygon(
                    [
                        (x - size, y - size),
                        (x + size, y - size),
                        (x, y)
                    ],
                    closed=True,
                    fill=False,
                    color="black"
                )
                ax.add_patch(triangle)

            # --- posuvná podpora (jen uy) ---
            elif fix_uy:
                
                triangle = patches.Polygon(
                    [
                        (x - size, y - size),
                        (x + size, y - size),
                        (x, y)
                    ],
                    closed=True,
                    fill=False,
                    edgecolor="black"
                )
                ax.add_patch(triangle)

                # kolečka
                circle1 = patches.Circle((x - size/2, y - size*1.3), size*0.2, fill=False)
                circle2 = patches.Circle((x + size/2, y - size*1.3), size*0.2, fill=False)
                circle3 = patches.Circle((x , y - size*1.3), size*0.2, fill=False)
                
                ax.add_patch(circle1)
                ax.add_patch(circle2)
                ax.add_patch(circle3)

                # základová čára
                ax.plot(
                    [x - 1.3*size, x + 1.3*size],
                    [y - 1.6*size, y - 1.6*size],
                    color="black",
                    linewidth=1
                )
            
            # --- posuvná podpora (jen ux) ---
            elif fix_ux:

                triangle = patches.Polygon(
                    [
                        (x - size, y + size),
                        (x - size, y - size),
                        (x, y)
                    ],
                    closed=True,
                    fill=False,
                    edgecolor="black"
                )
                ax.add_patch(triangle)

                # kolečka
                circle1 = patches.Circle((x - size*1.3, y + size/2), size*0.2, fill=False)
                circle2 = patches.Circle((x - size*1.3, y - size/2), size*0.2, fill=False)
                circle3 = patches.Circle((x - size*1.3, y ), size*0.2, fill=False)

                ax.add_patch(circle1)
                ax.add_patch(circle2)
                ax.add_patch(circle3)

                # základová čára
                ax.plot(
                    [x - 1.6*size, x - 1.6*size],
                    [y - 1.3*size, y + 1.3*size],
                    color="black",
                    linewidth=1
                )

    #Vykreslit vnitřní klouby
    def plot_releases(self, ax, scale=0.01):
        """
        Vykreslí momentové release (vnitřní klouby).

        Parametry:
            ax     : matplotlib axes
            scale  : relativní velikost symbolu vzhledem k rozměru konstrukce
        """

        # -------------------------------------------------
        # 1) Charakteristická velikost konstrukce
        # -------------------------------------------------

        xs = [node.x for node in self.nodes.values()]
        ys = [node.y for node in self.nodes.values()]

        Lchar = max(max(xs) - min(xs), max(ys) - min(ys))
        radius = scale * Lchar
        eps = 1.5 * radius

        # -------------------------------------------------
        # 2) Pro každý geometrický node najdi připojené prvky
        # -------------------------------------------------

        for node_id, node in self.nodes.items():

            connected = []  # (element, end_type, rz_dof)

            for elem in self.elements.values():

                dn_i = self.dof_nodes[elem.i]
                dn_j = self.dof_nodes[elem.j]

                if dn_i.node_id == node_id:
                    connected.append((elem, "i", dn_i.rz))

                if dn_j.node_id == node_id:
                    connected.append((elem, "j", dn_j.rz))

            if len(connected) < 2:
                continue  # jeden prut → není co řešit

            # -------------------------------------------------
            # 3) Seskup podle rotačního DOF
            # -------------------------------------------------

            rz_groups = defaultdict(list)

            for item in connected:
                elem, end_type, rz = item
                rz_groups[rz].append(item)

            if len(rz_groups) == 1:
                continue  # tuhý uzel

            # -------------------------------------------------
            # 4) Plný kloub (všechny skupiny mají 1 prvek)
            # -------------------------------------------------

            if all(len(group) == 1 for group in rz_groups.values()):

                circle = plt.Circle(
                    (node.x, node.y),
                    radius=radius,
                    facecolor="white",
                    edgecolor="black",
                    zorder=20,
                )
                ax.add_patch(circle)

            # -------------------------------------------------
            # 5) Částečný release → kresli excentricky
            # -------------------------------------------------

            else:

                for group in rz_groups.values():

                    # jen skupiny o velikosti 1 jsou release
                    if len(group) == 1:

                        elem, end_type, rz = group[0]

                        # získej druhý konec prvku
                        dn_i = self.dof_nodes[elem.i]
                        dn_j = self.dof_nodes[elem.j]

                        node_i = self.nodes[dn_i.node_id]
                        node_j = self.nodes[dn_j.node_id]

                        dx = node_j.x - node_i.x
                        dy = node_j.y - node_i.y
                        L = np.hypot(dx, dy)

                        if L == 0:
                            continue

                        ex = dx / L
                        ey = dy / L

                        if end_type == "i":
                            x_plot = node.x + ex * eps
                            y_plot = node.y + ey * eps
                        else:
                            x_plot = node.x - ex * eps
                            y_plot = node.y - ey * eps

                        circle = plt.Circle(
                            (x_plot, y_plot),
                            radius=radius,
                            facecolor="white",
                            edgecolor="black",
                            zorder=20,
                        )
                        ax.add_patch(circle)


def main():
    parser = argparse.ArgumentParser(
        description="2D rámový výpočet z JSON modelu."
    )
    parser.add_argument(
        "input_file",
        help="Cesta k vstupnímu JSON souboru (např. test_model.json).",
    )
    parser.add_argument(
        "--export-animation",
        dest="export_animation_path",
        help="Uloží vytvořenou animaci do souboru (.gif, .mp4, .mov, .m4v, .html).",
    )
    parser.add_argument(
        "--animation-view",
        choices=["dashboard", "response", "deformation", "N", "V", "M"],
        default="dashboard",
        help="Typ animace pro export v režimu 'Dynamic - steady state'.",
    )
    parser.add_argument(
        "--animation-writer",
        choices=["pillow", "ffmpeg", "html"],
        help="Vynutí writer pro export animace. Pokud není zadán, odvodí se z přípony souboru.",
    )
    parser.add_argument(
        "--animation-fps",
        type=float,
        default=20.0,
        help="Snímková frekvence exportované animace.",
    )
    parser.add_argument(
        "--animation-dpi",
        type=int,
        default=120,
        help="DPI pro rasterový export animace.",
    )
    args = parser.parse_args()

    model = Model.from_json(args.input_file)
    print(f"problem_type: {model.problem_type}")

    if model.problem_type == "Static":
        model.solve()
        print(model.compute_reactions())
        print(model.U)
        model.print_reactions()
        model.plot_deformed_shape()
        model.plot_internal_forces('M')
        model.plot_internal_forces('V')
        model.plot_internal_forces('N')
        model.show_all_plots()

    elif model.problem_type == "Dynamic - Natural frequencies and modes":
        model.solve_dynamic()
        model.print_dynamic_results()
        model.plot_all_mode_shapes(show=True)

    elif model.problem_type == "Dynamic - steady state":
        model.solve_dynamic_steady_state()
        model.print_dynamic_results()
        model.clear_animation_references()

        animation_view = args.animation_view.upper() if args.animation_view in {"N", "V", "M"} else args.animation_view
        show_animation = args.export_animation_path is None
        if animation_view == "dashboard":
            model.plot_dynamic_results_dashboard(show=show_animation)
        elif animation_view in ("response", "deformation"):
            model.plot_dynamic_response_animation(show=show_animation)
        else:
            model.plot_dynamic_internal_force_animation(kind=animation_view, show=show_animation)

        if args.export_animation_path:
            exported_path = model.export_last_animation(
                args.export_animation_path,
                writer=args.animation_writer,
                fps=args.animation_fps,
                dpi=args.animation_dpi,
            )
            print(f"Animation exported to: {exported_path}")
        elif show_animation:
            plt.show()

    elif model.problem_type == "Dynamic - FRF":
        model.solve_dynamic_frf()
        model.print_dynamic_results()
        model.plot_dynamic_frf(show=True)

    elif model.problem_type == "Stability":
        model.solve_stability()
        model.print_stability_results()
        model.plot_all_stability_shapes(show=True)

    else:
        print(f"Unsupported problem_type '{model.problem_type}', fallback to Static solve.")
        model.solve()
        print(model.compute_reactions())
        print(model.U)
        model.print_reactions()
        model.plot_deformed_shape()
        model.plot_internal_forces('M')
        model.plot_internal_forces('V')
        model.plot_internal_forces('N')
        model.show_all_plots()


if __name__ == '__main__':
    main()
