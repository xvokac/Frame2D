import json
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from Frame_2D import Model


class Frame2DGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Frame2D - editor a solver")

        self.nodes = {}
        self.dof_nodes = {}
        self.sections = {}
        self.elements = {}
        self.supports = []
        self.nodal_loads = []
        self.element_loads = []

        self.next_node_id = 1
        self.next_dof_node_id = 1
        self.next_section_id = 1
        self.next_element_id = 1
        self.next_dof_id = 1

        self.mode = None
        self.pending_element_nodes = []
        self.pending_release_element = None

        self.view_scale = 40.0
        self.view_origin_x = 80.0
        self.view_origin_y = 620.0

        self._build_ui()

    def _build_ui(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        ttk.Button(toolbar, text="Section", command=self.add_section_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Nodes", command=lambda: self.set_mode("add_node")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Elements", command=lambda: self.set_mode("add_element")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Supports", command=lambda: self.set_mode("add_support")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Nodal loads", command=lambda: self.set_mode("add_nodal_load")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Element load", command=lambda: self.set_mode("add_element_load")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Vnitřní kloub", command=lambda: self.set_mode("add_release_pick_element")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Editace zadání", command=self.open_data_editor).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Load JSON", command=self.load_json).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Uložit JSON", command=self.save_json).pack(side=tk.LEFT, padx=8)
        ttk.Button(toolbar, text="Výpočet + grafy", command=self.solve_and_plot).pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="Režim: žádný")
        ttk.Label(self.root, textvariable=self.status_var).pack(anchor="w", padx=10)

        self.canvas = tk.Canvas(self.root, width=1000, height=700, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel_linux)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel_linux)

        reactions_frame = ttk.LabelFrame(self.root, text="Výpis reakcí")
        reactions_frame.pack(fill=tk.BOTH, expand=False, padx=6, pady=(0, 6))

        self.reactions_text = tk.Text(reactions_frame, height=8, wrap=tk.NONE)
        self.reactions_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.reactions_text.insert("1.0", "Po výpočtu se zde zobrazí reakce.")
        self.reactions_text.configure(state=tk.DISABLED)

    def set_mode(self, mode: str):
        self.mode = mode
        self.pending_element_nodes = []
        self.pending_release_element = None
        self.status_var.set(f"Režim: {mode}")

    def to_canvas(self, x, y):
        return self.view_origin_x + x * self.view_scale, self.view_origin_y - y * self.view_scale

    def from_canvas(self, cx, cy):
        return (cx - self.view_origin_x) / self.view_scale, (self.view_origin_y - cy) / self.view_scale

    def zoom_at(self, cx, cy, factor):
        wx, wy = self.from_canvas(cx, cy)
        new_scale = self.view_scale * factor
        self.view_scale = max(8.0, min(300.0, new_scale))
        self.view_origin_x = cx - wx * self.view_scale
        self.view_origin_y = cy + wy * self.view_scale
        self.draw_scene()

    def on_mouse_wheel(self, event):
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        self.zoom_at(event.x, event.y, factor)

    def on_mouse_wheel_linux(self, event):
        factor = 1.1 if event.num == 4 else 1 / 1.1
        self.zoom_at(event.x, event.y, factor)

    def draw_scene(self):
        self.canvas.delete("all")

        for e in self.elements.values():
            ni = self.nodes[self.dof_nodes[e["i"]]["node"]]
            nj = self.nodes[self.dof_nodes[e["j"]]["node"]]
            xi, yi = self.to_canvas(ni["x"], ni["y"])
            xj, yj = self.to_canvas(nj["x"], nj["y"])
            self.canvas.create_line(xi, yi, xj, yj, fill="black", width=2)
            self._draw_release_symbol_if_needed(e, "i", xi, yi, xj, yj)
            self._draw_release_symbol_if_needed(e, "j", xj, yj, xi, yi)
            mx, my = (xi + xj) / 2, (yi + yj) / 2
            self.canvas.create_text(mx, my - 8, text=f"E{e['id']}", fill="darkgreen")

        for n_id, node in self.nodes.items():
            x, y = self.to_canvas(node["x"], node["y"])
            r = 4
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="royalblue")
            self.canvas.create_text(x + 12, y - 10, text=f"N{n_id}", fill="blue")

        self._draw_supports()
        self._draw_nodal_loads()
        self._draw_element_loads()

    def _draw_supports(self):
        for support in self.supports:
            dof_node = self.dof_nodes.get(support.get("node"))
            if not dof_node:
                continue
            node = self.nodes.get(dof_node["node"])
            if not node:
                continue
            x, y = self.to_canvas(node["x"], node["y"])
            ux = bool(support.get("ux"))
            uy = bool(support.get("uy"))
            rz = bool(support.get("rz"))

            if ux and uy and rz:
                self.canvas.create_rectangle(x - 8, y + 8, x + 8, y + 24, outline="firebrick", width=2)
            elif ux and uy:
                self.canvas.create_polygon(x, y + 8, x - 10, y + 24, x + 10, y + 24, outline="firebrick", fill="", width=2)
            elif uy:
                self.canvas.create_polygon(x, y + 8, x - 10, y + 20, x + 10, y + 20, outline="firebrick", fill="", width=2)
                self.canvas.create_oval(x - 9, y + 20, x - 3, y + 26, outline="firebrick", width=2)
                self.canvas.create_oval(x + 3, y + 20, x + 9, y + 26, outline="firebrick", width=2)
            else:
                self.canvas.create_text(x, y + 18, text="S", fill="firebrick")

    def _draw_arrow(self, x1, y1, x2, y2, color="tomato", width=2):
        self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width, arrow=tk.LAST)

    def _draw_nodal_loads(self):
        for load in self.nodal_loads:
            dof_node = self.dof_nodes.get(load.get("dof_node"))
            if not dof_node:
                continue
            node = self.nodes.get(dof_node["node"])
            if not node:
                continue

            x, y = self.to_canvas(node["x"], node["y"])
            fx = float(load.get("Fx", 0.0))
            fy = float(load.get("Fy", 0.0))
            mz = float(load.get("Mz", 0.0))

            if abs(fx) > 1e-12:
                sign = 1 if fx > 0 else -1
                self._draw_arrow(x - 18 * sign, y - 14, x + 18 * sign, y - 14)
                self.canvas.create_text(x, y - 26, text=f"Fx={fx:g}", fill="tomato", font=("TkDefaultFont", 8))

            if abs(fy) > 1e-12:
                sign = -1 if fy > 0 else 1
                self._draw_arrow(x + 14, y + 18 * sign, x + 14, y - 18 * sign)
                self.canvas.create_text(x + 38, y, text=f"Fy={fy:g}", fill="tomato", font=("TkDefaultFont", 8))

            if abs(mz) > 1e-12:
                r = 12
                self.canvas.create_arc(x - r, y - r, x + r, y + r, start=30, extent=300, style=tk.ARC, outline="tomato", width=2)
                if mz > 0:
                    self._draw_arrow(x + 8, y - 8, x + 10, y - 18)
                else:
                    self._draw_arrow(x + 10, y - 18, x + 8, y - 8)
                self.canvas.create_text(x - 26, y - 18, text=f"Mz={mz:g}", fill="tomato", font=("TkDefaultFont", 8))

    def _draw_element_loads(self):
        for load in self.element_loads:
            element = self.elements.get(load.get("element"))
            if not element:
                continue
            ni = self.nodes.get(self.dof_nodes.get(element["i"], {}).get("node"))
            nj = self.nodes.get(self.dof_nodes.get(element["j"], {}).get("node"))
            if not ni or not nj:
                continue

            x1, y1 = self.to_canvas(ni["x"], ni["y"])
            x2, y2 = self.to_canvas(nj["x"], nj["y"])
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-9:
                continue
            ux = dx / length
            uy = dy / length
            nx = -uy
            ny = ux

            qx = float(load.get("qx", 0.0))
            qz = float(load.get("qz", 0.0))

            if abs(qx) > 1e-12:
                sign = 1 if qx > 0 else -1
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                self._draw_arrow(mx - 22 * ux * sign, my - 22 * uy * sign, mx + 22 * ux * sign, my + 22 * uy * sign)
                self.canvas.create_text(mx + 10 * nx, my + 10 * ny, text=f"qx={qx:g}", fill="tomato", font=("TkDefaultFont", 8))

            if abs(qz) > 1e-12:
                sign = -1 if qz > 0 else 1
                for t in (0.2, 0.5, 0.8):
                    px = x1 + t * dx
                    py = y1 + t * dy
                    sx = px + 18 * nx * sign
                    sy = py + 18 * ny * sign
                    self._draw_arrow(sx, sy, px, py)
                tx = (x1 + x2) / 2 + 26 * nx * sign
                ty = (y1 + y2) / 2 + 26 * ny * sign
                self.canvas.create_text(tx, ty, text=f"qz={qz:g}", fill="tomato", font=("TkDefaultFont", 8))

    def _get_primary_dof_node_id(self, node_id):
        candidates = [did for did, d in self.dof_nodes.items() if d["node"] == node_id]
        if not candidates:
            return None
        return min(candidates)

    def _draw_release_symbol_if_needed(self, element, end_key, x, y, ox, oy):
        did = element[end_key]
        dof = self.dof_nodes.get(did)
        if not dof:
            return
        primary = self._get_primary_dof_node_id(dof["node"])
        if primary is None or did == primary:
            return
        vx = ox - x
        vy = oy - y
        length = math.hypot(vx, vy)
        if length < 1e-9:
            return
        ux = vx / length
        uy = vy / length
        cx = x + ux * 10
        cy = y + uy * 10
        r = 5
        self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="white", outline="purple", width=2)

    def find_nearest_node(self, cx, cy, tol=12):
        best = None
        best_d = 1e9
        for nid, node in self.nodes.items():
            x, y = self.to_canvas(node["x"], node["y"])
            d = math.hypot(cx - x, cy - y)
            if d < best_d:
                best_d = d
                best = nid
        return best if best_d <= tol else None

    def find_nearest_element(self, cx, cy, tol=10):
        best = None
        best_d = 1e9
        for eid, elem in self.elements.items():
            ni = self.nodes[self.dof_nodes[elem["i"]]["node"]]
            nj = self.nodes[self.dof_nodes[elem["j"]]["node"]]
            x1, y1 = self.to_canvas(ni["x"], ni["y"])
            x2, y2 = self.to_canvas(nj["x"], nj["y"])
            d = self._point_to_segment_distance(cx, cy, x1, y1, x2, y2)
            if d < best_d:
                best_d = d
                best = eid
        return best if best_d <= tol else None

    @staticmethod
    def _point_to_segment_distance(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        qx, qy = x1 + t * dx, y1 + t * dy
        return math.hypot(px - qx, py - qy)

    def add_section_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Nová section")

        labels = ["E", "A", "I"]
        entries = {}
        for i, lbl in enumerate(labels):
            ttk.Label(win, text=lbl).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            ent = ttk.Entry(win)
            ent.grid(row=i, column=1, padx=6, pady=4)
            entries[lbl] = ent

        def save():
            try:
                E = float(entries["E"].get())
                A = float(entries["A"].get())
                I = float(entries["I"].get())
            except ValueError:
                messagebox.showerror("Chyba", "E, A, I musí být čísla.")
                return

            sid = self.next_section_id
            self.next_section_id += 1
            self.sections[sid] = {"id": sid, "E": E, "A": A, "I": I}
            win.destroy()

        ttk.Button(win, text="Uložit", command=save).grid(row=4, column=0, columnspan=2, pady=8)

    def add_node_dialog(self, x, y):
        win = tk.Toplevel(self.root)
        win.title("Nový node")

        ttk.Label(win, text="x").grid(row=0, column=0, padx=6, pady=4)
        ex = ttk.Entry(win)
        ex.insert(0, f"{x:.3f}")
        ex.grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(win, text="y").grid(row=1, column=0, padx=6, pady=4)
        ey = ttk.Entry(win)
        ey.insert(0, f"{y:.3f}")
        ey.grid(row=1, column=1, padx=6, pady=4)

        def save():
            try:
                nx = float(ex.get())
                ny = float(ey.get())
            except ValueError:
                messagebox.showerror("Chyba", "Souřadnice musí být čísla.")
                return

            nid = self.next_node_id
            self.next_node_id += 1
            self.nodes[nid] = {"id": nid, "x": nx, "y": ny}

            dof_id = self.next_dof_node_id
            self.next_dof_node_id += 1
            ux = self.next_dof_id
            uy = self.next_dof_id + 1
            rz = self.next_dof_id + 2
            self.next_dof_id += 3
            self.dof_nodes[dof_id] = {"id": dof_id, "node": nid, "ux": ux, "uy": uy, "rz": rz}

            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Uložit", command=save).grid(row=3, column=0, columnspan=2, pady=8)

    def add_element_dialog(self, n1, n2):
        if not self.sections:
            messagebox.showwarning("Section", "Nejdřív zadej aspoň jednu section.")
            return

        win = tk.Toplevel(self.root)
        win.title("Nový element")
        ttk.Label(win, text=f"Node {n1} -> Node {n2}").grid(row=0, column=0, columnspan=2, padx=6, pady=4)

        ttk.Label(win, text="Section").grid(row=1, column=0, padx=6, pady=4)
        cb = ttk.Combobox(win, values=[str(sid) for sid in self.sections.keys()], state="readonly")
        cb.current(0)
        cb.grid(row=1, column=1, padx=6, pady=4)

        def save():
            sid = int(cb.get())
            eid = self.next_element_id
            self.next_element_id += 1

            dn1 = self.get_primary_dof_node_for_node(n1)
            dn2 = self.get_primary_dof_node_for_node(n2)
            self.elements[eid] = {"id": eid, "i": dn1, "j": dn2, "section": sid}
            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Uložit", command=save).grid(row=2, column=0, columnspan=2, pady=8)

    def get_primary_dof_node_for_node(self, node_id):
        for did, d in self.dof_nodes.items():
            if d["node"] == node_id:
                return did
        raise ValueError(f"DofNode pro node {node_id} neexistuje")

    def on_canvas_click(self, event):
        x, y = self.from_canvas(event.x, event.y)

        if self.mode == "add_node":
            self.add_node_dialog(x, y)

        elif self.mode == "add_element":
            n = self.find_nearest_node(event.x, event.y)
            if n is None:
                return
            self.pending_element_nodes.append(n)
            if len(self.pending_element_nodes) == 2:
                n1, n2 = self.pending_element_nodes
                self.pending_element_nodes = []
                if n1 == n2:
                    messagebox.showerror("Chyba", "Element musí mít 2 různé nody.")
                    return
                self.add_element_dialog(n1, n2)

        elif self.mode == "add_support":
            n = self.find_nearest_node(event.x, event.y)
            if n is not None:
                self.add_support_dialog(n)

        elif self.mode == "add_nodal_load":
            n = self.find_nearest_node(event.x, event.y)
            if n is not None:
                self.add_nodal_load_dialog(n)

        elif self.mode == "add_element_load":
            e = self.find_nearest_element(event.x, event.y)
            if e is not None:
                self.add_element_load_dialog(e)

        elif self.mode == "add_release_pick_element":
            e = self.find_nearest_element(event.x, event.y)
            if e is not None:
                self.pending_release_element = e
                self.mode = "add_release_pick_node"
                self.status_var.set(f"Režim: vyber konec elementu E{e}")

        elif self.mode == "add_release_pick_node":
            n = self.find_nearest_node(event.x, event.y)
            if n is not None and self.pending_release_element is not None:
                self.apply_release(self.pending_release_element, n)
                self.pending_release_element = None
                self.mode = "add_release_pick_element"
                self.status_var.set("Režim: add_release_pick_element")

    def add_support_dialog(self, node_id):
        win = tk.Toplevel(self.root)
        win.title(f"Support node {node_id}")

        ux = tk.BooleanVar(value=False)
        uy = tk.BooleanVar(value=True)
        rz = tk.BooleanVar(value=False)

        ttk.Checkbutton(win, text="ux", variable=ux).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(win, text="uy", variable=uy).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(win, text="rz", variable=rz).pack(anchor="w", padx=8, pady=2)

        def save():
            dn = self.get_primary_dof_node_for_node(node_id)
            self.supports.append({"node": dn, "ux": ux.get(), "uy": uy.get(), "rz": rz.get()})
            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Uložit", command=save).pack(pady=8)

    def add_nodal_load_dialog(self, node_id):
        win = tk.Toplevel(self.root)
        win.title(f"Nodal load node {node_id}")

        names = ["Fx", "Fy", "Mz"]
        entries = {}
        for i, nm in enumerate(names):
            ttk.Label(win, text=nm).grid(row=i, column=0, padx=6, pady=4)
            e = ttk.Entry(win)
            e.insert(0, "0")
            e.grid(row=i, column=1, padx=6, pady=4)
            entries[nm] = e

        def save():
            try:
                Fx = float(entries["Fx"].get())
                Fy = float(entries["Fy"].get())
                Mz = float(entries["Mz"].get())
            except ValueError:
                messagebox.showerror("Chyba", "Fx, Fy, Mz musí být čísla.")
                return

            dn = self.get_primary_dof_node_for_node(node_id)
            self.nodal_loads.append({"dof_node": dn, "Fx": Fx, "Fy": Fy, "Mz": Mz})
            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Uložit", command=save).grid(row=4, column=0, columnspan=2, pady=8)

    def add_element_load_dialog(self, element_id):
        win = tk.Toplevel(self.root)
        win.title(f"Element load E{element_id}")

        ttk.Label(win, text="qx").grid(row=0, column=0, padx=6, pady=4)
        ex = ttk.Entry(win)
        ex.insert(0, "0")
        ex.grid(row=0, column=1, padx=6, pady=4)

        ttk.Label(win, text="qz").grid(row=1, column=0, padx=6, pady=4)
        ez = ttk.Entry(win)
        ez.insert(0, "0")
        ez.grid(row=1, column=1, padx=6, pady=4)

        def save():
            try:
                qx = float(ex.get())
                qz = float(ez.get())
            except ValueError:
                messagebox.showerror("Chyba", "qx, qz musí být čísla.")
                return

            self.element_loads.append({"element": element_id, "qx": qx, "qz": qz})
            win.destroy()
            self.draw_scene()

        ttk.Button(win, text="Uložit", command=save).grid(row=2, column=0, columnspan=2, pady=8)

    def apply_release(self, element_id, node_id):
        elem = self.elements[element_id]
        did_i = elem["i"]
        did_j = elem["j"]

        end = None
        if self.dof_nodes[did_i]["node"] == node_id:
            end = "i"
            base = self.dof_nodes[did_i]
        elif self.dof_nodes[did_j]["node"] == node_id:
            end = "j"
            base = self.dof_nodes[did_j]
        else:
            messagebox.showerror("Kloub", "Vybraný node není na konci elementu.")
            return

        new_did = self.next_dof_node_id
        self.next_dof_node_id += 1
        new_rz = self.next_dof_id + 2
        self.next_dof_id += 3
        self.dof_nodes[new_did] = {
            "id": new_did,
            "node": node_id,
            "ux": base["ux"],
            "uy": base["uy"],
            "rz": new_rz,
        }

        if end == "i":
            elem["i"] = new_did
        else:
            elem["j"] = new_did

        self.draw_scene()

    def build_json_model(self):
        return {
            "nodes": list(self.nodes.values()),
            "dof_nodes": list(self.dof_nodes.values()),
            "sections": list(self.sections.values()),
            "elements": list(self.elements.values()),
            "supports": self.supports,
            "nodal_loads": self.nodal_loads,
            "element_loads": self.element_loads,
        }

    def save_json(self):
        path = filedialog.asksaveasfilename(
            title="Uložit model", defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return

        data = self.build_json_model()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        messagebox.showinfo("Uloženo", f"Model uložen do:\n{path}")

    def load_json(self):
        path = filedialog.askopenfilename(title="Načíst model", filetypes=[("JSON", "*.json")])
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.nodes = {n["id"]: n for n in data.get("nodes", [])}
            self.dof_nodes = {d["id"]: {"id": d["id"], "node": d["node"], "ux": d["ux"], "uy": d["uy"], "rz": d["rz"]} for d in data.get("dof_nodes", [])}
            self.sections = {s["id"]: s for s in data.get("sections", [])}
            self.elements = {
                e["id"]: {"id": e["id"], "i": e["i"], "j": e["j"], "section": e["section"]}
                for e in data.get("elements", [])
            }
            self.supports = data.get("supports", [])
            self.nodal_loads = data.get("nodal_loads", [])
            self.element_loads = data.get("element_loads", [])

            self.next_node_id = max(self.nodes.keys(), default=0) + 1
            self.next_dof_node_id = max(self.dof_nodes.keys(), default=0) + 1
            self.next_section_id = max(self.sections.keys(), default=0) + 1
            self.next_element_id = max(self.elements.keys(), default=0) + 1

            max_dof = 0
            for d in self.dof_nodes.values():
                max_dof = max(max_dof, d["ux"], d["uy"], d["rz"])
            self.next_dof_id = max_dof + 1

            self.draw_scene()
            messagebox.showinfo("Načteno", f"Model načten z:\n{path}")
        except Exception as exc:
            messagebox.showerror("Load JSON", f"Nepodařilo se načíst model:\n{exc}")

    def solve_and_plot(self):
        if not self.elements or not self.nodes:
            messagebox.showwarning("Solver", "Model musí mít aspoň nody a elementy.")
            return

        tmp_path = "_frame2d_gui_temp.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.build_json_model(), f, indent=2)

        try:
            model = Model.from_json(tmp_path)
            model.solve()
            self.show_reactions(model.format_reactions())
            model.plot_deformed_shape()
            model.plot_internal_forces("M")
            model.plot_internal_forces("V")
            model.plot_internal_forces("N")
            model.show_all_plots()
        except Exception as exc:
            messagebox.showerror("Solver chyba", str(exc))

    def show_reactions(self, text):
        self.reactions_text.configure(state=tk.NORMAL)
        self.reactions_text.delete("1.0", tk.END)
        self.reactions_text.insert("1.0", text)
        self.reactions_text.configure(state=tk.DISABLED)

    def open_data_editor(self):
        editor = tk.Toplevel(self.root)
        editor.title("Editace zadání")
        editor.geometry("980x560")

        notebook = ttk.Notebook(editor)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        schemas = {
            "sections": ["id", "E", "A", "I"],
            "nodes": ["id", "x", "y"],
            "elements": ["id", "i", "j", "section"],
            "supports": ["node", "ux", "uy", "rz"],
            "nodal loads": ["dof_node", "Fx", "Fy", "Mz"],
            "element load": ["element", "qx", "qz"],
            "release rotation": ["element", "end", "node", "dof_node"],
        }

        data_tables = {
            "sections": [dict(v) for v in self.sections.values()],
            "nodes": [dict(v) for v in self.nodes.values()],
            "elements": [dict(v) for v in self.elements.values()],
            "supports": [dict(v) for v in self.supports],
            "nodal loads": [dict(v) for v in self.nodal_loads],
            "element load": [dict(v) for v in self.element_loads],
            "release rotation": self._collect_release_rows(),
        }

        trees = {}

        for tab_name, columns in schemas.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=tab_name)
            tree = ttk.Treeview(frame, columns=columns, show="headings", height=14)
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=110, anchor="center")
            tree.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
            trees[tab_name] = tree

            btns = ttk.Frame(frame)
            btns.pack(fill=tk.X, pady=6)
            if tab_name == "release rotation":
                ttk.Label(btns, text="Tabulka je pouze informativní (odvozeno z elementů + dof_nodes).").pack(side=tk.LEFT, padx=4)
            else:
                ttk.Button(btns, text="Přidat řádek", command=lambda n=tab_name: self._add_table_row(n, schemas[n], data_tables, trees)).pack(side=tk.LEFT, padx=4)
                ttk.Button(btns, text="Upravit řádek", command=lambda n=tab_name: self._edit_table_row(n, schemas[n], data_tables, trees)).pack(side=tk.LEFT, padx=4)
                ttk.Button(btns, text="Smazat řádek", command=lambda n=tab_name: self._delete_table_row(n, data_tables, trees)).pack(side=tk.LEFT, padx=4)

            self._refresh_tree(tree, columns, data_tables[tab_name])

        def apply_changes():
            try:
                self._apply_editor_tables(data_tables)
                self.draw_scene()
                editor.destroy()
                messagebox.showinfo("Editace", "Data byla aktualizována.")
            except Exception as exc:
                messagebox.showerror("Editace", str(exc))

        ttk.Button(editor, text="Použít změny", command=apply_changes).pack(pady=(0, 8))

    def _collect_release_rows(self):
        rows = []
        for element in self.elements.values():
            for end in ("i", "j"):
                did = element[end]
                dof = self.dof_nodes.get(did)
                if not dof:
                    continue
                primary = self._get_primary_dof_node_id(dof["node"])
                if primary is not None and did != primary:
                    rows.append({"element": element["id"], "end": end, "node": dof["node"], "dof_node": did})
        return rows

    def _refresh_tree(self, tree, columns, rows):
        tree.delete(*tree.get_children())
        for idx, row in enumerate(rows):
            tree.insert("", tk.END, iid=str(idx), values=[row.get(c, "") for c in columns])

    def _edit_row_dialog(self, title, columns, initial=None):
        win = tk.Toplevel(self.root)
        win.title(title)
        entries = {}
        initial = initial or {}
        for i, col in enumerate(columns):
            ttk.Label(win, text=col).grid(row=i, column=0, padx=6, pady=4, sticky="w")
            ent = ttk.Entry(win)
            ent.insert(0, str(initial.get(col, "")))
            ent.grid(row=i, column=1, padx=6, pady=4)
            entries[col] = ent

        result = {"row": None}

        def save_row():
            row = {col: entries[col].get().strip() for col in columns}
            result["row"] = row
            win.destroy()

        ttk.Button(win, text="Uložit", command=save_row).grid(row=len(columns), column=0, columnspan=2, pady=8)
        win.transient(self.root)
        win.grab_set()
        self.root.wait_window(win)
        return result["row"]

    def _add_table_row(self, tab_name, columns, data_tables, trees):
        row = self._edit_row_dialog(f"Přidat: {tab_name}", columns)
        if row is None:
            return
        data_tables[tab_name].append(row)
        self._refresh_tree(trees[tab_name], columns, data_tables[tab_name])

    def _edit_table_row(self, tab_name, columns, data_tables, trees):
        tree = trees[tab_name]
        sel = tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        row = self._edit_row_dialog(f"Upravit: {tab_name}", columns, data_tables[tab_name][idx])
        if row is None:
            return
        data_tables[tab_name][idx] = row
        self._refresh_tree(tree, columns, data_tables[tab_name])

    def _delete_table_row(self, tab_name, data_tables, trees):
        tree = trees[tab_name]
        sel = tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        data_tables[tab_name].pop(idx)
        columns = tree["columns"]
        self._refresh_tree(tree, columns, data_tables[tab_name])

    @staticmethod
    def _to_bool(value):
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _apply_editor_tables(self, data_tables):
        sections = {
            int(r["id"]): {"id": int(r["id"]), "E": float(r["E"]), "A": float(r["A"]), "I": float(r["I"])}
            for r in data_tables["sections"]
        }
        nodes = {int(r["id"]): {"id": int(r["id"]), "x": float(r["x"]), "y": float(r["y"])} for r in data_tables["nodes"]}

        elements = {}
        for r in data_tables["elements"]:
            eid = int(r["id"])
            elements[eid] = {"id": eid, "i": int(r["i"]), "j": int(r["j"]), "section": int(r["section"])}

        supports = [
            {
                "node": int(r["node"]),
                "ux": self._to_bool(r["ux"]),
                "uy": self._to_bool(r["uy"]),
                "rz": self._to_bool(r["rz"]),
            }
            for r in data_tables["supports"]
        ]
        nodal_loads = [
            {
                "dof_node": int(r["dof_node"]),
                "Fx": float(r["Fx"]),
                "Fy": float(r["Fy"]),
                "Mz": float(r["Mz"]),
            }
            for r in data_tables["nodal loads"]
        ]
        element_loads = [
            {"element": int(r["element"]), "qx": float(r["qx"]), "qz": float(r["qz"])}
            for r in data_tables["element load"]
        ]

        missing_sections = [e["id"] for e in elements.values() if e["section"] not in sections]
        if missing_sections:
            raise ValueError(f"Elementy odkazují na neexistující section: {missing_sections}")

        unknown_dof_in_elements = sorted({did for e in elements.values() for did in (e["i"], e["j"]) if did not in self.dof_nodes})
        if unknown_dof_in_elements:
            raise ValueError(f"Elementy odkazují na neexistující dof_node: {unknown_dof_in_elements}")

        unknown_dof_in_supports = sorted({s["node"] for s in supports if s["node"] not in self.dof_nodes})
        if unknown_dof_in_supports:
            raise ValueError(f"Supports odkazují na neexistující dof_node: {unknown_dof_in_supports}")

        unknown_dof_in_nodal_loads = sorted({l["dof_node"] for l in nodal_loads if l["dof_node"] not in self.dof_nodes})
        if unknown_dof_in_nodal_loads:
            raise ValueError(f"Nodal loads odkazují na neexistující dof_node: {unknown_dof_in_nodal_loads}")

        unknown_elements_in_loads = sorted({l["element"] for l in element_loads if l["element"] not in elements})
        if unknown_elements_in_loads:
            raise ValueError(f"Element load odkazuje na neexistující element: {unknown_elements_in_loads}")

        # release rotation tab je odvozený pohled; změny zde se zatím neaplikují zpět
        self.sections = sections
        self.nodes = nodes
        self.elements = elements
        self.supports = supports
        self.nodal_loads = nodal_loads
        self.element_loads = element_loads

        self.next_section_id = max(self.sections.keys(), default=0) + 1
        self.next_node_id = max(self.nodes.keys(), default=0) + 1
        self.next_element_id = max(self.elements.keys(), default=0) + 1
        self.next_dof_node_id = max(self.dof_nodes.keys(), default=0) + 1
        max_dof = max((max(d["ux"], d["uy"], d["rz"]) for d in self.dof_nodes.values()), default=0)
        self.next_dof_id = max_dof + 1


def main():
    root = tk.Tk()
    app = Frame2DGui(root)
    app.draw_scene()
    root.mainloop()


if __name__ == "__main__":
    main()
