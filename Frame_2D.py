## ODM

import numpy as np
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict



    
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
        self.U = []
        self.dof_map = {}
        self.dynamic_dofs = []
        self.eigenvalues = np.array([])
        self.eigenvectors = np.array([[]])

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

        model.normalize_ids()

        return model

    def normalize_ids(self):
        """Přečísluje ID na souvislé řady (1..n), aby byly konzistentní pro solver."""

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

        if not self.dynamic_dofs:
            raise ValueError("Dynamic analysis requires at least one active DOF with positive mass.")

        idx = [self.dof_map[dof] for dof in self.dynamic_dofs]
        K_red = K[np.ix_(idx, idx)]

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

        return self.eigenvalues, self.eigenvectors, reduced_dofs

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
        for i, dof in enumerate(self.dynamic_dofs):
            U_mode[dof - 1] = mode_vec[i]

        return U_mode

    def format_dynamic_results(self):
        if len(self.eigenvalues) == 0:
            return "No dynamic results available."

        lines = ["DYNAMIC EIGEN SOLUTION:"]
        lines.append(f"Reduced DOFs: {self.dynamic_dofs}")

        for i, lam in enumerate(self.eigenvalues, start=1):
            lines.append(f"lambda_{i} = {lam:.6e}")
            mode = self.eigenvectors[:, i - 1]
            lines.append(f"U_{i} = {mode}")

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
        ax.set_aspect('equal')
        ax.grid(True)
        ax.margins(0.2)
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

        omega = np.sqrt(max(self.eigenvalues[mode_index], 0.0))
        freq_hz = omega / (2 * np.pi)
        ax.set_title(f"Model with reduced DOFs - Shape mode no. {mode_index + 1}: f = {freq_hz:.3f} Hz")
        ax.set_aspect('equal')
        ax.grid(True)
        ax.margins(0.2)

        if show:
            plt.show()

    def plot_all_mode_shapes(self, show=False):
        n_modes = min(self.number_of_eigenvectors, self.eigenvectors.shape[1] if self.eigenvectors.ndim == 2 else 0)
        for i in range(n_modes):
            self.plot_mode_shape(i, show=False)
        if show:
            self.show_all_plots()

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

                
    #koncové síly na prvku
    def element_end_forces(self, elem):
        sec = self.sections[elem.section_id]
        L, alpha = self.element_geometry(elem.id)
        # lokální tuhost — VOLÁ SE FUNKCE MIMO TŘÍDU
        k_local = local_stiffness(sec.E, sec.A, sec.I, L)        
        # transformace
        T = transformation_matrix(alpha)
        # globální posuny prvku
        u_global = self.get_element_displacements(elem)
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

    def plot_internal_forces(self, kind="M", scale=None, show=False):
        if scale is None:
            scale = self.auto_scale(kind)
        plt.figure()
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
        plt.axis("equal")
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


if __name__ == '__main__':
    model = Model.from_json('test_dyn_03.json')
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
