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


def extract_minutiae(skeleton_255: np.ndarray) -> Tuple[List[Minutia], List[Minutia]]:
    """Extrae terminaciones y bifurcaciones usando CN."""
    img = to_binary01(skeleton_255)
    h, w = img.shape
    endings: List[Minutia] = []
    bifurcations: List[Minutia] = []

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if img[y, x] != 1:
                continue
            cn = crossing_number(img, y, x)
            if cn == 1:
                endings.append(Minutia(x=x, y=y, kind="ending", cn=cn))
            elif cn == 3:
                bifurcations.append(Minutia(x=x, y=y, kind="bifurcation", cn=cn))

    return endings, bifurcations


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
    endings: List[Minutia],
    bifurcations: List[Minutia],
) -> np.ndarray:
    """Dibuja minucias sobre huella adelgazada (BGR)."""
    overlay = cv2.cvtColor(skeleton_255, cv2.COLOR_GRAY2BGR)

    for m in endings:
        cv2.circle(overlay, (m.x, m.y), 3, (255, 0, 0), 1)  # azul
    for m in bifurcations:
        cv2.circle(overlay, (m.x, m.y), 3, (0, 0, 255), 1)  # rojo

    cv2.putText(
        overlay,
        "Terminaciones: {} | Bifurcaciones: {}".format(len(endings), len(bifurcations)),
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return overlay


def main() -> None:
    gray = load_fingerprint_tif(INPUT_TIF_PATH)
    binary = preprocess_and_binarize(gray)

    skeleton = thin_image(binary)
    pruned = prune_skeleton(skeleton, min_branch_length=PRUNE_MIN_BRANCH_LENGTH)

    endings, bifurcations = extract_minutiae(pruned)
    endings = filter_minutiae(
        endings,
        img_shape=pruned.shape,
        border_margin=MINUTIAE_BORDER_MARGIN,
        min_distance=MINUTIAE_MIN_DISTANCE,
    )
    bifurcations = filter_minutiae(
        bifurcations,
        img_shape=pruned.shape,
        border_margin=MINUTIAE_BORDER_MARGIN,
        min_distance=MINUTIAE_MIN_DISTANCE,
    )

    result = draw_minutiae_overlay(pruned, endings, bifurcations)

    print("Pipeline completado")
    print("Terminaciones detectadas: {}".format(len(endings)))
    print("Bifurcaciones detectadas: {}".format(len(bifurcations)))

    if DISPLAY_SCALE > 1.0:
        display_img = cv2.resize(
            result,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        display_img = result

    cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_WINDOW_NAME, display_img.shape[1], display_img.shape[0])
    cv2.imshow(DISPLAY_WINDOW_NAME, display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

