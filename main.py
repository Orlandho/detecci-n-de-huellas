import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


# Ruta fija del archivo TIFF de huella.
INPUT_TIF_PATH = "huella.tif"

# Parametros del pipeline.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
MORPH_KERNEL_SIZE = 3
MORPH_CLOSE_ITERS = 1
MAX_THIN_ITERATIONS = 50
PRUNE_MIN_BRANCH_LENGTH = 12
MINUTIAE_BORDER_MARGIN = 10
MINUTIAE_MIN_DISTANCE = 10
DISPLAY_SCALE = 2.0
DISPLAY_WINDOW_NAME = "Huella adelgazada + minucias"
SELECTED_MINUTIA_TYPE = "all"  # "ending", "bifurcation" o "all"
DRAW_MINUTIA_INDEX = True


@dataclass
class Minutia:
    x: int
    y: int
    kind: str
    cn: int


def load_fingerprint_tif(path: str) -> np.ndarray:
    """Lee una huella .tif con PIL y la devuelve como matriz uint8 en gris."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontro el archivo TIFF en: {path}. "
            "Actualiza INPUT_TIF_PATH con una ruta valida."
        )
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def preprocess_and_binarize(gray: np.ndarray) -> np.ndarray:
    """Mejora contraste, binariza e intenta dejar crestas en blanco (255)."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = float(np.count_nonzero(binary)) / float(binary.size)
    # Las crestas suelen ocupar menos area que el fondo; si no, invertimos.
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERS)
    return cleaned


def to_binary01(img_255: np.ndarray) -> np.ndarray:
    return (img_255 > 0).astype(np.uint8)


def to_binary255(img_01: np.ndarray) -> np.ndarray:
    return img_01.astype(np.uint8) * 255


def zhang_suen_thinning(binary_255: np.ndarray, max_iterations: int = 50) -> np.ndarray:
    """Fallback manual de thinning por vecindarios 3x3 (Zhang-Suen)."""
    img = to_binary01(binary_255)
    rows, cols = img.shape

    for _ in range(max_iterations):
        removed_any = False

        to_remove = []
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if img[y, x] != 1:
                    continue
                p2 = img[y - 1, x]
                p3 = img[y - 1, x + 1]
                p4 = img[y, x + 1]
                p5 = img[y + 1, x + 1]
                p6 = img[y + 1, x]
                p7 = img[y + 1, x - 1]
                p8 = img[y, x - 1]
                p9 = img[y - 1, x - 1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]

                b = int(sum(neighbors))
                if b < 2 or b > 6:
                    continue

                sequence = neighbors + [neighbors[0]]
                a = sum((sequence[i] == 0 and sequence[i + 1] == 1) for i in range(8))
                if a != 1:
                    continue

                if p2 * p4 * p6 != 0:
                    continue
                if p4 * p6 * p8 != 0:
                    continue

                to_remove.append((y, x))

        if to_remove:
            removed_any = True
            for y, x in to_remove:
                img[y, x] = 0

        to_remove = []
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if img[y, x] != 1:
                    continue
                p2 = img[y - 1, x]
                p3 = img[y - 1, x + 1]
                p4 = img[y, x + 1]
                p5 = img[y + 1, x + 1]
                p6 = img[y + 1, x]
                p7 = img[y + 1, x - 1]
                p8 = img[y, x - 1]
                p9 = img[y - 1, x - 1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]

                b = int(sum(neighbors))
                if b < 2 or b > 6:
                    continue

                sequence = neighbors + [neighbors[0]]
                a = sum((sequence[i] == 0 and sequence[i + 1] == 1) for i in range(8))
                if a != 1:
                    continue

                if p2 * p4 * p8 != 0:
                    continue
                if p2 * p6 * p8 != 0:
                    continue

                to_remove.append((y, x))

        if to_remove:
            removed_any = True
            for y, x in to_remove:
                img[y, x] = 0

        if not removed_any:
            break

    return to_binary255(img)


def thin_image(binary_255: np.ndarray) -> np.ndarray:
    """Thinning: usa cv2.ximgproc.thinning si existe, con fallback manual."""
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "thinning"):
        return ximgproc.thinning(binary_255)
    return zhang_suen_thinning(binary_255, max_iterations=MAX_THIN_ITERATIONS)


def get_neighbors8(img_01: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    h, w = img_01.shape
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and img_01[ny, nx] == 1:
                neighbors.append((ny, nx))
    return neighbors


def crossing_number(img_01: np.ndarray, y: int, x: int) -> int:
    """Numero de cruce CN en vecindario 3x3 para clasificar minucias."""
    p2 = img_01[y - 1, x]
    p3 = img_01[y - 1, x + 1]
    p4 = img_01[y, x + 1]
    p5 = img_01[y + 1, x + 1]
    p6 = img_01[y + 1, x]
    p7 = img_01[y + 1, x - 1]
    p8 = img_01[y, x - 1]
    p9 = img_01[y - 1, x - 1]
    ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
    transitions = sum(abs(int(ring[i]) - int(ring[i + 1])) for i in range(8))
    return transitions // 2


def trace_branch(
    img_01: np.ndarray,
    start: Tuple[int, int],
    max_steps: int,
) -> List[Tuple[int, int]]:
    """Recorre una rama partiendo de un endpoint hasta bifurcacion/final."""
    path = [start]
    prev = None
    current = start

    for _ in range(max_steps):
        neigh = get_neighbors8(img_01, current[0], current[1])
        if prev is not None:
            neigh = [pt for pt in neigh if pt != prev]

        if not neigh:
            break

        # Si hay mas de un vecino, llegamos a una bifurcacion.
        if len(neigh) > 1:
            break

        nxt = neigh[0]
        if crossing_number(img_01, nxt[0], nxt[1]) >= 3:
            break

        path.append(nxt)
        prev, current = current, nxt

        if crossing_number(img_01, current[0], current[1]) == 1 and current != start:
            break

    return path


def prune_skeleton(skeleton_255: np.ndarray, min_branch_length: int) -> np.ndarray:
    """Poda ramas cortas (spurs) detectadas desde endpoints."""
    img = to_binary01(skeleton_255)
    h, w = img.shape
    changed = True

    while changed:
        changed = False
        endpoints = []

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if img[y, x] == 1 and crossing_number(img, y, x) == 1:
                    endpoints.append((y, x))

        if not endpoints:
            break

        for ep in endpoints:
            if img[ep[0], ep[1]] == 0:
                continue
            path = trace_branch(img, ep, max_steps=min_branch_length + 2)
            if len(path) <= min_branch_length:
                for py, px in path:
                    img[py, px] = 0
                changed = True

    return to_binary255(img)


def extract_advanced_minutiae(skeleton_255: np.ndarray, min_branch_length: int = 12) -> List[Minutia]:
    img = to_binary01(skeleton_255)
    h, w = img.shape

    nodes = {}
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if img[y, x] == 1:
                cn = crossing_number(img, y, x)
                if cn != 2:
                    nodes[(y, x)] = cn

    visited_directed_edges = set()
    edges = []

    for node in list(nodes.keys()):
        cn = nodes[node]
        if cn == 0:
            continue
        neighbors = get_neighbors8(img, node[0], node[1])
        for nxt in neighbors:
            if (node, nxt) in visited_directed_edges:
                continue

            path = [node, nxt]
            prev = node
            curr = nxt

            while True:
                if curr[0] <= 0 or curr[0] >= h - 1 or curr[1] <= 0 or curr[1] >= w - 1:
                    break

                if curr in nodes and curr != node:
                    break

                curr_cn = crossing_number(img, curr[0], curr[1])
                if curr_cn != 2 and curr not in nodes:
                    nodes[curr] = curr_cn
                    break

                if curr in nodes and curr == node:
                    break

                neighs = get_neighbors8(img, curr[0], curr[1])
                next_steps = [n for n in neighs if n != prev]

                if not next_steps:
                    break

                prev = curr
                curr = next_steps[0]

                if curr in path:
                    break

                path.append(curr)

            end_node = curr
            if len(path) >= 2:
                visited_directed_edges.add((path[0], path[1]))
                visited_directed_edges.add((path[-1], path[-2]))

            edges.append({
                'n1': node,
                'n2': end_node,
                'path': path,
                'length': len(path)
            })

    minutiae_list = []
    used_nodes = set()

    # 1. Islands (cn=0)
    for node, cn in nodes.items():
        if cn == 0:
            minutiae_list.append(Minutia(x=node[1], y=node[0], kind="island", cn=0))
            used_nodes.add(node)

    edge_pairs = {}
    for e in edges:
        pair = tuple(sorted([e['n1'], e['n2']]))
        if pair not in edge_pairs:
            edge_pairs[pair] = []
        edge_pairs[pair].append(e)

    # 2. Lakes
    for pair, edgs in edge_pairs.items():
        if len(edgs) >= 2:
            n1, n2 = pair
            cn1, cn2 = nodes.get(n1, 2), nodes.get(n2, 2)
            if cn1 >= 3 and cn2 >= 3:
                mid_idx = len(edgs[0]['path']) // 2
                mid = edgs[0]['path'][mid_idx]
                minutiae_list.append(Minutia(x=mid[1], y=mid[0], kind="lake", cn=3))
                used_nodes.add(n1)
                used_nodes.add(n2)
                for e in edgs:
                    e['used'] = True

        if len(edgs) == 1 and pair[0] == pair[1]:
            n1 = pair[0]
            cn1 = nodes.get(n1, 2)
            if cn1 >= 3:
                mid_idx = len(edgs[0]['path']) // 2
                mid = edgs[0]['path'][mid_idx]
                minutiae_list.append(Minutia(x=mid[1], y=mid[0], kind="lake", cn=3))
                used_nodes.add(n1)
                edgs[0]['used'] = True

    # 3. Independent Ridge & Island, Spur, Cross over
    for e in edges:
        if e.get('used'):
            continue
        n1, n2 = e['n1'], e['n2']
        cn1, cn2 = nodes.get(n1, 2), nodes.get(n2, 2)
        l = e['length']

        if cn1 == 1 and cn2 == 1:
            mid = e['path'][l // 2]
            if l <= 5:
                minutiae_list.append(Minutia(x=mid[1], y=mid[0], kind="island", cn=1))
            elif l <= 30:
                minutiae_list.append(Minutia(x=mid[1], y=mid[0], kind="independ rige", cn=1))
            used_nodes.add(n1)
            used_nodes.add(n2)
            e['used'] = True

        elif (cn1 == 1 and cn2 >= 3) or (cn1 >= 3 and cn2 == 1):
            if l <= min_branch_length:
                spur_node = n1 if cn1 == 1 else n2
                minutiae_list.append(Minutia(x=spur_node[1], y=spur_node[0], kind="spur", cn=1))
                used_nodes.add(spur_node)
                e['used'] = True

        elif cn1 >= 3 and cn2 >= 3:
            if l <= min_branch_length:
                mid = e['path'][l // 2]
                minutiae_list.append(Minutia(x=mid[1], y=mid[0], kind="cross over", cn=3))
                used_nodes.add(n1)
                used_nodes.add(n2)
                e['used'] = True

    # 4. Endings and Bifurcations
    for node, cn in nodes.items():
        if node not in used_nodes:
            if cn == 1:
                minutiae_list.append(Minutia(x=node[1], y=node[0], kind="terminacion", cn=1))
            elif cn >= 3:
                minutiae_list.append(Minutia(x=node[1], y=node[0], kind="bifurcacion", cn=3))

    return minutiae_list


def filter_minutiae(
    minutiae: List[Minutia],
    img_shape: Tuple[int, int],
    border_margin: int,
    min_distance: int,
) -> List[Minutia]:
    """Filtra minucias por borde y por distancia minima entre puntos."""
    h, w = img_shape

    valid = [
        m
        for m in minutiae
        if border_margin <= m.x < (w - border_margin)
        and border_margin <= m.y < (h - border_margin)
    ]

    selected: List[Minutia] = []
    min_dist_sq = float(min_distance * min_distance)

    for m in sorted(valid, key=lambda q: (q.y, q.x)):
        keep = True
        for s in selected:
            dx = float(m.x - s.x)
            dy = float(m.y - s.y)
            if dx * dx + dy * dy < min_dist_sq:
                keep = False
                break
        if keep:
            selected.append(m)

    return selected


def draw_minutiae_overlay(
    skeleton_255: np.ndarray,
    minutiae: List[Minutia],
    selected_type: str,
) -> np.ndarray:
    """Dibuja minucias sobre huella adelgazada (BGR)."""
    overlay = cv2.cvtColor(skeleton_255, cv2.COLOR_GRAY2BGR)

    # Colores diferentes para cada tipo
    color_map = {
        "terminacion": (255, 0, 0),       # Blue
        "bifurcacion": (0, 0, 255),       # Red
        "lake": (0, 255, 0),              # Green
        "independ rige": (0, 255, 255),   # Yellow
        "island": (255, 255, 0),          # Cyan
        "spur": (255, 0, 255),            # Magenta
        "cross over": (128, 0, 128)       # Purple
    }

    for idx, m in enumerate(minutiae, start=1):
        color = color_map.get(m.kind, (255, 255, 255))
        cv2.circle(overlay, (m.x, m.y), 3, color, 1)
        if DRAW_MINUTIA_INDEX:
            cv2.putText(
                overlay,
                str(idx),
                (m.x + 4, m.y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        overlay,
        "Tipo: {} | Cantidad: {}".format(selected_type, len(minutiae)),
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return overlay


def select_minutiae_by_type(
    minutiae: List[Minutia],
    selected_type: str,
) -> List[Minutia]:
    selected = selected_type.strip().lower()
    if selected == "all":
        return minutiae
    return [m for m in minutiae if m.kind == selected]


import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

def main() -> None:
    # 1. Preprocesamiento base (se hace una sola vez)
    gray = load_fingerprint_tif(INPUT_TIF_PATH)
    binary = preprocess_and_binarize(gray)
    skeleton = thin_image(binary)

    all_minutiae = extract_advanced_minutiae(skeleton, min_branch_length=PRUNE_MIN_BRANCH_LENGTH)
    all_minutiae = filter_minutiae(
        all_minutiae,
        img_shape=skeleton.shape,
        border_margin=MINUTIAE_BORDER_MARGIN,
        min_distance=MINUTIAE_MIN_DISTANCE,
    )

    # Contar minucias
    counts = {}
    for m in all_minutiae:
        counts[m.kind] = counts.get(m.kind, 0) + 1

    # Crear ventana de Tkinter
    root = tk.Tk()
    root.title("Extractor de Minucias")

    # Frame izquierdo para controles
    frame_left = tk.Frame(root, padx=10, pady=10)
    frame_left.pack(side=tk.LEFT, fill=tk.Y)

    # Frame derecho para imagen
    frame_right = tk.Frame(root, padx=10, pady=10)
    frame_right.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    tk.Label(frame_left, text="Tipo de Minucia:", font=("Helvetica", 12, "bold")).pack(pady=(0, 5))

    tipos = ["all", "terminacion", "bifurcacion", "lake", "independ rige", "island", "spur", "cross over"]
    combo = ttk.Combobox(frame_left, values=tipos, state="readonly")
    combo.current(0)
    combo.pack(pady=5)

    stats_label = tk.Label(frame_left, text="", justify=tk.LEFT, anchor="w", font=("Helvetica", 10))
    stats_label.pack(pady=20, fill=tk.X)

    img_label = tk.Label(frame_right)
    img_label.pack(expand=True, fill=tk.BOTH)

    def update_display(event=None):
        selected_type = combo.get()
        selected_minutiae = select_minutiae_by_type(all_minutiae, selected_type)

        # Dibujar imagen
        result_bgr = draw_minutiae_overlay(skeleton, selected_minutiae, selected_type)
        if DISPLAY_SCALE > 1.0:
            result_bgr = cv2.resize(
                result_bgr,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_NEAREST,
            )

        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(result_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        img_label.config(image=img_tk)
        img_label.image = img_tk  # mantener referencia

        # Actualizar estadisticas
        stats_text = "Estadísticas Globales:\n"
        for t in tipos[1:]:
            stats_text += f"{t.capitalize()}: {counts.get(t, 0)}\n"
        stats_text += f"Total: {len(all_minutiae)}\n\n"
        stats_text += f"Mostrando: {len(selected_minutiae)}"

        stats_label.config(text=stats_text)

    combo.bind("<<ComboboxSelected>>", update_display)

    # Inicializar pantalla
    update_display()

    root.mainloop()


if __name__ == "__main__":
    main()

