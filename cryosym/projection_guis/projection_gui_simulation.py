import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QSizePolicy, QGroupBox, QCheckBox, QFrame,
    QPushButton, QProgressBar, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import logging
import pickle
import os

from cryosym.volume_download.data_downloader import data_downloader
from aspire.volume import Volume
from aspire.utils.rotation import Rotation

from cryosym.group_elements import group_elements, normalizer, coset_representatives
from cryosym.utils import multi_Jify
from cryosym.config import ROTATIONS_CACHE_DIR

logger = logging.getLogger(__name__)


def get_pi_rotation_x():
    """Get the rotation matrix for π rotation about the x-axis"""
    # R_x(π) = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    return np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float64)


def get_z_rotation(angle):
    """Get the rotation matrix for rotation about the z-axis by given angle (in radians)"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=np.float64)


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.ax = self.fig.add_subplot(111)

    def plot_image(self, image_data, title="", bottom_text="", count_text=""):
        self.ax.clear()
        if image_data is not None:
            self.ax.imshow(image_data, cmap='gray')
            self.ax.set_title(title, fontsize=9)
            self.ax.axis('off')

            if bottom_text:
                self.ax.text(0.5, -0.05, bottom_text, ha='center', va='top',
                             transform=self.ax.transAxes, fontsize=8, color='blue')

            if count_text:
                self.ax.text(0.5, -0.25, count_text, ha='center', va='top',
                             transform=self.ax.transAxes, fontsize=8, color='green')
        else:
            self.ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_title(title, fontsize=10)
        self.draw()

    def plot_projections(self, projections, title="", max_images=5):
        self.ax.clear()
        if projections is not None:
            try:
                if hasattr(projections, 'asnumpy'):
                    proj_array = projections.asnumpy()
                elif hasattr(projections, '__array__'):
                    proj_array = np.array(projections)
                else:
                    proj_array = projections

                if hasattr(proj_array, 'shape') and len(proj_array.shape) >= 3:
                    n_images = proj_array.shape[0]
                elif hasattr(projections, 'n'):
                    n_images = projections.n
                    proj_array = projections.asnumpy()
                else:
                    n_images = len(proj_array) if hasattr(proj_array, '__len__') else 1

                if n_images > 0:
                    n_show = min(n_images, max_images)
                    cols = min(3, n_show)
                    rows = (n_show + cols - 1) // cols

                    self.fig.clear()
                    for i in range(n_show):
                        ax = self.fig.add_subplot(rows, cols, i + 1)
                        if len(proj_array.shape) >= 3:
                            ax.imshow(proj_array[i], cmap='gray')
                        else:
                            ax.imshow(proj_array, cmap='gray')
                            break
                        ax.set_title(f'Proj {i}', fontsize=8)
                        ax.axis('off')
                    self.fig.suptitle(title, fontsize=10)
                else:
                    raise ValueError("No images to display")

            except Exception as e:
                ax = self.fig.add_subplot(111)
                ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
                ax.set_title(title, fontsize=10)
        else:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10)

        self.fig.tight_layout()
        self.draw()

    def plot_histogram(self, data, title="", xlabel="", ylabel="", bins=30, color='skyblue', alpha=0.7):
        """Plot a histogram of the given data"""
        self.ax.clear()
        if data is not None and len(data) > 0:
            clean_data = data[np.isfinite(data)]
            if len(clean_data) > 0:
                bin_edges = [0, 2, 4, 6, 8, 10, np.inf]
                bin_labels = ['0-2°', '2-4°', '4-6°', '6-8°', '8-10°', '10+°']

                counts, _ = np.histogram(clean_data, bins=bin_edges)

                bar_positions = []
                bar_widths = []
                for i in range(len(bin_edges) - 1):
                    if bin_edges[i + 1] == np.inf:
                        bar_positions.append(12)  # Position for '10+' bin
                        bar_widths.append(4)  # Width for '10+' bin
                    else:
                        center = (bin_edges[i] + bin_edges[i + 1]) / 2
                        width = bin_edges[i + 1] - bin_edges[i]
                        bar_positions.append(center)
                        bar_widths.append(width)

                bars = self.ax.bar(bar_positions, counts, width=bar_widths, color=color, alpha=alpha,
                                   edgecolor='black', align='center')

                self.ax.set_xticks(bar_positions)
                self.ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)

                self.ax.tick_params(axis='y', labelsize=8)

                self.ax.set_title(title, fontsize=9)
                self.ax.set_xlabel(xlabel, fontsize=8)
                self.ax.set_ylabel(ylabel, fontsize=8)
                self.ax.grid(True, alpha=0.3, axis='y')

                mean_val = np.mean(clean_data)
                std_val = np.std(clean_data)
                median_val = np.median(clean_data)
                stats_text = f'Mean: {mean_val:.2f}°\nMedian: {median_val:.2f}°\nStd: {std_val:.2f}°'
                self.ax.text(0.98, 0.98, stats_text, transform=self.ax.transAxes,
                             verticalalignment='top', horizontalalignment='right',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                             fontsize=7)

                for bar, count in zip(bars, counts):
                    if count > 0:
                        height = bar.get_height()
                        self.ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(counts),
                                     f'{int(count)}', ha='center', va='bottom', fontsize=7)

            else:
                self.ax.text(0.5, 0.5, 'No valid data to plot', ha='center', va='center',
                             transform=self.ax.transAxes)
                self.ax.set_title(title, fontsize=9)
        else:
            self.ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                         transform=self.ax.transAxes)
            self.ax.set_title(title, fontsize=10)
        self.draw()


class DataProcessor(QThread):
    """Thread for processing volume data without blocking the GUI"""
    data_ready = pyqtSignal(object, object, object, object, object)
    progress_update = pyqtSignal(str)

    def __init__(self, sym='T', cache_file=None, simulation_rotations=None, cache_selected_data=None, img_size=70,
                 num_imgs=12):
        super().__init__()
        self.sym = sym
        self.cache_file = cache_file
        self.simulation_rotations = simulation_rotations
        self.cache_selected_data = cache_selected_data
        self.img_size = img_size
        self.num_imgs = num_imgs

    def run(self):
        try:
            self.progress_update.emit("Loading rotation cache...")
            cache_data = None
            R = None

            if self.cache_file and (isinstance(self.cache_file, str) and os.path.exists(self.cache_file) or (hasattr(self.cache_file, 'exists') and self.cache_file.exists())):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if isinstance(cache_data, (list, tuple)) and len(cache_data) >= 1:
                        R = cache_data[0]
                        self.progress_update.emit(f"Loaded {len(R)} cached rotations")
                    else:
                        raise ValueError("Invalid cache file format")
            else:
                self.progress_update.emit("Cache file not found or not provided, creating dummy cache...")
                R = np.random.rand(100, 3, 3)  # Dummy cache of 100 rotations
                cache_data = [R, None, None, None, None]

            self.progress_update.emit("Loading volume...")

            try:
                volume_file = data_downloader(self.sym)
                og_vol = Volume.load(volume_file)
                og_vol = og_vol.downsample(250)
            except Exception as e:
                self.progress_update.emit(f"Volume load failed ({e}). Using mock volume.")
                og_vol = Volume.load("mock_volume.mrc")  # Use mock loader

            processed_sim_rotations = None
            if self.simulation_rotations is not None:
                self.progress_update.emit("Processing simulation rotations...")
                if isinstance(self.simulation_rotations, np.ndarray):
                    processed_sim_rotations = self.simulation_rotations
                elif hasattr(self.simulation_rotations, '__len__'):
                    processed_sim_rotations = np.array(self.simulation_rotations)
                else:
                    processed_sim_rotations = np.random.rand(50, 3, 3)

                self.progress_update.emit(f"Loaded {len(processed_sim_rotations)} simulation rotations")

            processed_cache_selected_data = None
            if self.cache_selected_data is not None:
                self.progress_update.emit("Processing cache selected data...")
                if isinstance(self.cache_selected_data, dict):
                    processed_cache_selected_data = self.cache_selected_data
                    if 'cache_selected_inds' in self.cache_selected_data:
                        self.progress_update.emit(
                            f"Loaded {len(self.cache_selected_data['cache_selected_inds'])} cache selected indices")
                    if 'cache_J_indices' in self.cache_selected_data:
                        self.progress_update.emit(
                            f"Loaded {len(self.cache_selected_data['cache_J_indices'])} cache J indices")
                else:
                    processed_cache_selected_data = None

            self.progress_update.emit("Processing complete!")
            self.data_ready.emit(og_vol, R, cache_data, processed_sim_rotations, processed_cache_selected_data)

        except Exception as e:
            self.progress_update.emit(f"Error: {str(e)}")
            dummy_rotations = np.random.rand(50, 3, 3)
            dummy_volume = Volume.load("mock_volume.mrc")  # Use mock loader
            dummy_cache = [dummy_rotations, None, None, None, None]
            dummy_sim_rotations = np.random.rand(25, 3, 3) if self.simulation_rotations is not None else None
            dummy_cache_selected_data = None
            if self.cache_selected_data is not None:
                dummy_cache_selected_data = {
                    "cache_selected_inds": np.random.randint(0, 50, 25),
                    "cache_J_indices": np.random.choice([-1, 1], 25),
                    "cache_selected_count": np.random.randint(1, 10, 25),
                    "cache_selected_r_indices": np.random.choice([-1, 1], 25)
                }
            self.data_ready.emit(dummy_volume, dummy_rotations, dummy_cache, dummy_sim_rotations,
                                 dummy_cache_selected_data)


class ProjectionViewer(QWidget):
    def __init__(self, sym='T', cache_file=None, simulation_rotations=None, cache_selected_data=None):
        super().__init__()
        self.sym = sym
        self.cache_file = cache_file
        self.simulation_rotations = simulation_rotations
        self.cache_selected_data = cache_selected_data

        # Check if this is C_n symmetry
        self.is_cn = isinstance(sym, str) and len(sym) > 1 and sym[0] == 'C'

        self.setWindowTitle(f"Projection Viewer (Simulation) - {sym} Symmetry")
        self.setMinimumSize(1250, 750)

        # Data storage
        self.volume = None
        self.cache_rotations = None
        self.cache_data = None
        self.sim_rotations = None
        self.current_rotation_index = 0
        self.current_sim_rotation_index = 0
        self.current_sim_rotation_selected = False
        self.closest_cache_indices = []

        # D2 Specifics
        # This index is for VISUAL INSPECTION (Center Panel) only
        self.current_cache_coset_index = 0
        # NEW: This index is for GLOBAL calculation (Histogram/Center Plot)
        self.global_coset_multiplier_index = 0  # Default to 0 (identity)

        self.coset_reps = None
        self.coset_names = None
        if sym == "D2":
            self.coset_reps = coset_representatives(sym)
            self.coset_names = ["1 (identity)", "ρ", "ρ²", "ε", "ρε", "ρ²ε"]
            self.coset_names_short = ["1", "ρ", "ρ²", "ε", "ρε", "ρ²ε"]

        # C_n Specifics: π rotation about x-axis
        self.pi_x_rotation = get_pi_rotation_x()
        # For visual inspection in center panel
        self.current_cache_pi_x_applied = False
        # For global calculation (histogram)
        self.global_pi_x_applied = False

        # C_n Specifics: K value for z-axis rotations
        self.cn_order = 1  # n from C_n
        if self.is_cn:
            try:
                self.cn_order = int(sym[1:])
            except ValueError:
                self.cn_order = 1
        self.K_value = 1  # K parameter: rotations are 2πk/(n*K) for k = 0, ..., n*K-1
        # For visual inspection in center panel
        self.current_cache_k_index = 0
        # For global calculation (histogram) - stores the k index for each rotation
        self.global_k_index = 0

        self.init_ui()
        self.load_data()

    def calculate_matrix_mse(self, matrix1, matrix2):
        """Calculate MSE between two rotation matrices"""
        if matrix1 is None or matrix2 is None:
            return None

        try:
            if matrix1.shape != matrix2.shape:
                return None

            mse = np.inf
            for g in group_elements(self.sym):
                mse_g = np.mean((matrix1 - g @ matrix2) ** 2)
                if mse_g < mse:
                    mse = mse_g
            return mse
        except Exception as e:
            print(f"Error calculating MSE: {e}")
            return None

    def calculate_angular_distance(self, matrix1, matrix2):
        """Calculate angular distance between two rotation matrices"""
        if matrix1 is None or matrix2 is None:
            return None

        try:
            if matrix1.shape != matrix2.shape:
                return None

            ang_dist = np.inf
            for g in group_elements(self.sym):
                ang_dist_g = Rotation.angle_dist(matrix1, g @ matrix2)
                if ang_dist_g < ang_dist:
                    ang_dist = ang_dist_g
            return ang_dist * 180 / np.pi
        except Exception as e:
            print(f"Error calculating angular distance: {e}")
            return None

    def calculate_all_angular_differences(self):
        """Calculate angular differences between all simulation rotations and their corresponding cache rotations"""
        if (self.sim_rotations is None or self.cache_rotations is None or
                self.cache_selected_data is None or 'cache_selected_inds' not in self.cache_selected_data):
            return None

        try:
            cache_selected_inds = self.cache_selected_data['cache_selected_inds']
            cache_J_indices = self.cache_selected_data.get('cache_J_indices', np.ones(len(cache_selected_inds)))

            # For D2, use coset indices
            if self.sym == "D2":
                cache_coset_indices = self.cache_selected_data.get('cache_coset_indices',
                                                                   np.zeros(len(cache_selected_inds), dtype=int))
                # NEW: Get the global multiplier
                global_coset_rep = self.coset_reps[self.global_coset_multiplier_index]
            # For C_n symmetry, use pi_x indices and k indices
            elif self.is_cn:
                cache_pi_x_indices = self.cache_selected_data.get('cache_pi_x_indices',
                                                                  np.ones(len(cache_selected_inds)))
                cache_k_indices = self.cache_selected_data.get('cache_k_indices',
                                                               np.zeros(len(cache_selected_inds), dtype=int))
            # For other D/T, use r indices
            else:
                cache_r_indices = self.cache_selected_data.get('cache_selected_r_indices',
                                                               np.ones(len(cache_selected_inds)))

            angular_differences = []

            for i in range(min(len(self.sim_rotations), len(cache_selected_inds))):
                # IMPORTANT: Sim rotation is ALWAYS the raw, unmodified ground truth
                sim_rotation = self.sim_rotations[i]
                cache_idx = cache_selected_inds[i]

                if cache_idx < len(self.cache_rotations):
                    cache_rotation = self.cache_rotations[cache_idx].copy()

                    # Apply D2 coset representative from DATA
                    if self.sym == "D2" and self.coset_reps is not None:
                        # 1. Apply individual coset rep from data
                        individual_coset_idx = cache_coset_indices[i]
                        individual_coset_rep = self.coset_reps[individual_coset_idx]
                        cache_rotation = individual_coset_rep @ cache_rotation

                        # 2. Apply global coset multiplier
                        cache_rotation = global_coset_rep @ cache_rotation

                    # Apply π rotation about x-axis for C_n symmetry (from the left)
                    elif self.is_cn:
                        # 1. Apply individual pi_x from data
                        if cache_pi_x_indices[i] == -1:
                            cache_rotation = self.pi_x_rotation @ cache_rotation

                        # 2. Apply global pi_x multiplier
                        if self.global_pi_x_applied:
                            cache_rotation = self.pi_x_rotation @ cache_rotation

                        # 3. Apply individual k rotation from data (right multiplication)
                        individual_k = cache_k_indices[i]
                        if individual_k != 0:
                            angle = 2 * np.pi * individual_k / (self.cn_order * self.K_value)
                            cache_rotation = cache_rotation @ get_z_rotation(angle)

                        # 4. Apply global k rotation (right multiplication)
                        if self.global_k_index != 0:
                            angle = 2 * np.pi * self.global_k_index / (self.cn_order * self.K_value)
                            cache_rotation = cache_rotation @ get_z_rotation(angle)

                    # Apply r for non-D2 T/D symmetries from DATA
                    elif (self.sym == "T" or (
                            isinstance(self.sym, str) and self.sym.startswith("D") and self.sym != "D2")):
                        if cache_r_indices[i] == -1:
                            normalizer_group = group_elements(normalizer(self.sym))
                            r = normalizer_group[-1]
                            cache_rotation = r @ cache_rotation

                    # Apply J
                    if cache_J_indices[i] == -1:
                        cache_rotation = multi_Jify(cache_rotation)

                    ang_dist = self.calculate_angular_distance(cache_rotation, sim_rotation)
                    if ang_dist is not None:
                        angular_differences.append(ang_dist)

            return np.array(angular_differences) if angular_differences else None

        except Exception as e:
            print(f"Error calculating angular differences: {e}")
            return None

    def update_angular_histogram(self):
        """Update the angular difference histogram"""
        angular_differences = self.calculate_all_angular_differences()

        if angular_differences is not None and len(angular_differences) > 0:
            self.histogram_plot.plot_histogram(
                angular_differences,
                title="Simulation vs Cache Rotations",
                xlabel="Angular Difference (degrees)",
                ylabel="Frequency",
                bins=30,
                color='skyblue'
            )
            self.progress_label.setText(f"Histogram updated with {len(angular_differences)} angular differences")
        else:
            self.histogram_plot.ax.clear()
            self.histogram_plot.ax.text(0.5, 0.5, 'No data available\nfor histogram',
                                        ha='center', va='center',
                                        transform=self.histogram_plot.ax.transAxes)
            self.histogram_plot.ax.set_title("Simulation vs Cache Rotations", fontsize=10)
            self.histogram_plot.draw()
        self.update_less_more_10_indices()

    def update_less_more_10_indices(self):
        """Update the list of indices for angular differences."""
        angular_differences = self.calculate_all_angular_differences()

        if angular_differences is not None:
            threshold = 10
            max_indices_in_view = 50

            less_than_10_indices = [i for i, diff in enumerate(angular_differences) if diff < threshold]
            more_than_10_indices = [i for i, diff in enumerate(angular_differences) if diff >= threshold]
            less_count = len(less_than_10_indices)
            more_count = len(more_than_10_indices)
            less_than_10_indices = less_than_10_indices[:max_indices_in_view]
            more_than_10_indices = more_than_10_indices[:max_indices_in_view]

            self.less_than_10_list.setText(
                f"<font color='skyblue'>{less_count} indices:</font><br>"
                f"{', '.join(map(str, less_than_10_indices)) or ''}")

            self.more_than_10_list.setText(
                f"<font color='skyblue'>{more_count} indices:</font><br>"
                f"{', '.join(map(str, more_than_10_indices)) or ''}"
            )
        else:
            self.less_than_10_list.setText("No Data")
            self.more_than_10_list.setText("No Data")

    def find_closest_cache_rotations(self, target_rotation, n_closest=3):
        """Find the n closest cache rotations to the target rotation, considering all transformations"""
        # This function is unchanged, as it finds the absolute best match,
        # independent of the global multiplier UI setting.
        if self.cache_rotations is None or target_rotation is None:
            return []

        distances = []

        # For D2, iterate over coset representatives instead of using r
        if self.sym == "D2" and self.coset_reps is not None:
            coset_indices = range(len(self.coset_reps))
        else:
            coset_indices = [0]  # No coset iteration for other symmetries

        # For C_n, iterate over pi_x values and k values
        if self.is_cn:
            pi_x_vals = [1, -1]  # 1 = no pi_x, -1 = apply pi_x
            k_vals = list(range(self.cn_order * self.K_value))  # k = 0, 1, ..., n*K-1
        else:
            pi_x_vals = [1]  # No pi_x for other symmetries
            k_vals = [0]  # No k rotation for other symmetries

        for i, cache_rot in enumerate(self.cache_rotations):
            for coset_idx in coset_indices:
                for t_val in [1]:  # 1 = no transpose, -1 = transpose
                    for j_val in [1, -1]:  # 1 = no J, -1 = apply J
                        for pi_x_val in pi_x_vals:
                            for k_val in k_vals:
                                # For non-D2 symmetries with r transformation
                                if self.sym == "D2":
                                    r_vals = [1]  # No r for D2 (coset handles this)
                                elif self.is_cn:
                                    r_vals = [1]  # No r for C_n (pi_x handles the extra transformation)
                                elif self.sym == "T" or (
                                        isinstance(self.sym, str) and self.sym.startswith("D") and self.sym != "D2"):
                                    r_vals = [1, -1]
                                else:
                                    r_vals = [1]

                                for r_val in r_vals:
                                    transformed_matrix = cache_rot.copy()

                                    # Apply D2 coset representative first (replaces r functionality)
                                    if self.sym == "D2" and self.coset_reps is not None:
                                        coset_rep = self.coset_reps[coset_idx]
                                        transformed_matrix = coset_rep @ transformed_matrix

                                    if t_val == -1:
                                        transformed_matrix = transformed_matrix.T

                                    # Apply π rotation about x-axis for C_n (from the left)
                                    if self.is_cn and pi_x_val == -1:
                                        transformed_matrix = self.pi_x_rotation @ transformed_matrix

                                    # Apply k rotation about z-axis for C_n (from the right)
                                    if self.is_cn and k_val != 0:
                                        angle = 2 * np.pi * k_val / (self.cn_order * self.K_value)
                                        transformed_matrix = transformed_matrix @ get_z_rotation(angle)

                                    # Apply r for non-D2 symmetries
                                    if r_val == -1 and self.sym != "D2" and not self.is_cn and (
                                            self.sym == "T" or (isinstance(self.sym, str) and self.sym.startswith("D"))):
                                        normalizer_group = group_elements(normalizer(self.sym))
                                        r = normalizer_group[-1]
                                        transformed_matrix = r @ transformed_matrix

                                    if j_val == -1:
                                        transformed_matrix = multi_Jify(transformed_matrix)

                                    ang_dist = self.calculate_angular_distance(transformed_matrix, target_rotation)

                                    if ang_dist is not None:
                                        distances.append((i, ang_dist, t_val, j_val, r_val, coset_idx, pi_x_val, k_val))

        distances.sort(key=lambda x: x[1])
        return distances[:n_closest]

    def init_ui(self):
        # Main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # =========================================
        # ==== LEFT PANEL (Simulation - LOCKED) ====
        # =========================================
        left_layout = QVBoxLayout()
        left_panel = QWidget()
        left_panel.setMaximumWidth(280)
        left_panel.setLayout(left_layout)

        sim_title = QLabel("Simulation (Ground Truth)")
        sim_title.setFont(QFont("Arial", 12, QFont.Bold))
        sim_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(sim_title)

        self.sim_rotation_count_label = QLabel("Total simulation rotations:\nLoading...")
        self.sim_rotation_count_label.setFont(QFont("Arial", 10))
        self.sim_rotation_count_label.setAlignment(Qt.AlignCenter)
        self.sim_rotation_count_label.setWordWrap(True)
        left_layout.addWidget(self.sim_rotation_count_label)

        sim_index_layout = QHBoxLayout()
        sim_index_label = QLabel("Rotation Index:")
        self.sim_index_input = QLineEdit()
        self.sim_index_input.setPlaceholderText("e.g., 0")
        self.sim_index_input.returnPressed.connect(self.update_simulation_rotation)

        sim_update_button = QPushButton("Show")
        sim_update_button.clicked.connect(self.update_simulation_rotation)

        sim_index_layout.addWidget(sim_index_label)
        sim_index_layout.addWidget(self.sim_index_input)
        sim_index_layout.addWidget(sim_update_button)
        left_layout.addLayout(sim_index_layout)

        self.left_plot = PlotCanvas(self, width=3, height=3)
        left_layout.addWidget(self.left_plot)

        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        self.progress_label.setWordWrap(True)
        self.progress_label.setMaximumHeight(50)
        left_layout.addWidget(self.progress_label)
        left_layout.addWidget(self.progress_bar)

        main_layout.addWidget(left_panel)

        # =========================================
        # ==== CENTER PANEL (Cache & Controls) ====
        # =========================================
        center_layout = QVBoxLayout()
        center_layout.setSpacing(10)

        center_panel = QWidget()
        center_panel.setMaximumWidth(300)
        center_panel.setLayout(center_layout)

        self.rotation_count_label = QLabel("Available cache rotations:\nLoading...")
        self.rotation_count_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.rotation_count_label.setAlignment(Qt.AlignCenter)
        self.rotation_count_label.setWordWrap(True)
        center_layout.addWidget(self.rotation_count_label)

        index_label = QLabel("Enter Cache Rotation Index:")
        index_label.setFont(QFont("Arial", 10))
        self.index_input = QLineEdit()
        self.index_input.setPlaceholderText("e.g., 3")
        self.index_input.returnPressed.connect(self.update_selected_rotation)

        update_button = QPushButton("Show Rotation")
        update_button.clicked.connect(self.update_selected_rotation)

        center_layout.addWidget(index_label)
        center_layout.addWidget(self.index_input)
        center_layout.addWidget(update_button)

        # --- Transformation Controls ---
        controls_box = QGroupBox("Transformation Controls")
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(8)

        # J value checkbox and flip button
        j_layout = QHBoxLayout()
        self.j_value_checkbox = QCheckBox("J (handedness)")
        self.j_value_checkbox.setFont(QFont("Arial", 10))

        self.flip_j_button = QPushButton("Flip All J")
        self.flip_j_button.clicked.connect(self.flip_j_values)
        self.flip_j_button.setMaximumWidth(80)

        j_layout.addWidget(self.j_value_checkbox)
        j_layout.addStretch()
        j_layout.addWidget(self.flip_j_button)
        controls_layout.addLayout(j_layout)

        # R value checkbox and flip button (NOT shown for D2 or C_n)
        r_layout = QHBoxLayout()
        self.r_value_checkbox = QCheckBox("r (normalizer coset)")
        self.r_value_checkbox.setFont(QFont("Arial", 10))

        self.flip_r_button = QPushButton("Flip All r")
        self.flip_r_button.clicked.connect(self.flip_r_values)
        self.flip_r_button.setMaximumWidth(80)

        r_layout.addWidget(self.r_value_checkbox)
        r_layout.addStretch()
        r_layout.addWidget(self.flip_r_button)
        controls_layout.addLayout(r_layout)

        self.r_value_checkbox.setVisible(False)
        self.flip_r_button.setVisible(False)

        # NEW: Add D2 specific controls
        if self.sym == "D2":
            # 1. Visual Inspection Control
            visual_coset_layout = QHBoxLayout()
            visual_coset_label = QLabel("View with Coset:")
            visual_coset_label.setFont(QFont("Arial", 9))
            self.cache_coset_combo = QComboBox()
            self.cache_coset_combo.addItems(self.coset_names)
            self.cache_coset_combo.currentIndexChanged.connect(self.on_cache_coset_view_changed)

            visual_coset_layout.addWidget(visual_coset_label)
            visual_coset_layout.addWidget(self.cache_coset_combo)
            controls_layout.addLayout(visual_coset_layout)

            # 2. NEW: Global Multiplier Control
            bulk_coset_group = QGroupBox("Global Coset Multiplier")
            bulk_layout = QVBoxLayout()
            bulk_layout.setSpacing(4)

            h_layout = QHBoxLayout()
            self.global_coset_combo = QComboBox()
            self.global_coset_combo.addItems(self.coset_names_short)  # Use short names

            self.global_coset_button = QPushButton("Apply Global")
            self.global_coset_button.setToolTip(
                "Applies this coset to ALL rotations\nfor histogram calculation and viewing.")
            self.global_coset_button.setStyleSheet("background-color: #dbeafe")  # Light blue
            self.global_coset_button.clicked.connect(self.apply_global_coset)

            h_layout.addWidget(QLabel("Apply Global:"))
            h_layout.addWidget(self.global_coset_combo)
            h_layout.addWidget(self.global_coset_button)

            bulk_layout.addLayout(h_layout)
            bulk_coset_group.setLayout(bulk_layout)
            controls_layout.addWidget(bulk_coset_group)

        # NEW: Add C_n specific controls
        if self.is_cn:
            # 1. Visual Inspection Control - checkbox for pi_x
            pi_x_layout = QHBoxLayout()
            self.pi_x_checkbox = QCheckBox("R_x(π) (left multiply)")
            self.pi_x_checkbox.setFont(QFont("Arial", 10))
            self.pi_x_checkbox.setToolTip("Apply π rotation about x-axis from the left")
            self.pi_x_checkbox.stateChanged.connect(self.on_pi_x_view_changed)

            self.flip_pi_x_button = QPushButton("Flip All π_x")
            self.flip_pi_x_button.clicked.connect(self.flip_pi_x_values)
            self.flip_pi_x_button.setMaximumWidth(80)

            pi_x_layout.addWidget(self.pi_x_checkbox)
            pi_x_layout.addStretch()
            pi_x_layout.addWidget(self.flip_pi_x_button)
            controls_layout.addLayout(pi_x_layout)

            # 2. Global π_x Multiplier Control
            bulk_pi_x_group = QGroupBox("Global π_x Multiplier")
            bulk_pi_x_layout = QVBoxLayout()
            bulk_pi_x_layout.setSpacing(4)

            h_layout_pi_x = QHBoxLayout()
            self.global_pi_x_checkbox = QCheckBox("Apply R_x(π) globally")
            self.global_pi_x_checkbox.setToolTip(
                "Applies R_x(π) to ALL rotations\nfor histogram calculation and viewing.")

            self.global_pi_x_button = QPushButton("Apply Global")
            self.global_pi_x_button.setStyleSheet("background-color: #dbeafe")  # Light blue
            self.global_pi_x_button.clicked.connect(self.apply_global_pi_x)

            h_layout_pi_x.addWidget(self.global_pi_x_checkbox)
            h_layout_pi_x.addWidget(self.global_pi_x_button)

            bulk_pi_x_layout.addLayout(h_layout_pi_x)
            bulk_pi_x_group.setLayout(bulk_pi_x_layout)
            controls_layout.addWidget(bulk_pi_x_group)

            # 3. K value input for z-axis rotations
            k_group = QGroupBox(f"Z-axis Rotation (n={self.cn_order})")
            k_group_layout = QVBoxLayout()
            k_group_layout.setSpacing(4)

            # K input row
            k_input_layout = QHBoxLayout()
            k_label = QLabel("K value:")
            k_label.setFont(QFont("Arial", 9))
            self.k_value_input = QLineEdit()
            self.k_value_input.setPlaceholderText("1")
            self.k_value_input.setText("1")
            self.k_value_input.setMaximumWidth(50)
            self.k_value_input.setToolTip(f"Rotations: 2πk/({self.cn_order}·K) for k=0..{self.cn_order}·K-1")

            self.k_apply_button = QPushButton("Apply K")
            self.k_apply_button.setStyleSheet("background-color: #dbeafe")
            self.k_apply_button.clicked.connect(self.apply_k_value)

            k_input_layout.addWidget(k_label)
            k_input_layout.addWidget(self.k_value_input)
            k_input_layout.addWidget(self.k_apply_button)
            k_group_layout.addLayout(k_input_layout)

            # Visual k selector
            visual_k_layout = QHBoxLayout()
            visual_k_label = QLabel("View with k:")
            visual_k_label.setFont(QFont("Arial", 9))
            self.visual_k_combo = QComboBox()
            self.visual_k_combo.addItems([str(i) for i in range(self.cn_order * self.K_value)])
            self.visual_k_combo.currentIndexChanged.connect(self.on_visual_k_changed)

            visual_k_layout.addWidget(visual_k_label)
            visual_k_layout.addWidget(self.visual_k_combo)
            k_group_layout.addLayout(visual_k_layout)

            # Global k selector
            global_k_layout = QHBoxLayout()
            global_k_label = QLabel("Global k:")
            global_k_label.setFont(QFont("Arial", 9))
            self.global_k_combo = QComboBox()
            self.global_k_combo.addItems([str(i) for i in range(self.cn_order * self.K_value)])

            self.global_k_button = QPushButton("Apply")
            self.global_k_button.setStyleSheet("background-color: #dbeafe")
            self.global_k_button.clicked.connect(self.apply_global_k)

            global_k_layout.addWidget(global_k_label)
            global_k_layout.addWidget(self.global_k_combo)
            global_k_layout.addWidget(self.global_k_button)
            k_group_layout.addLayout(global_k_layout)

            k_group.setLayout(k_group_layout)
            controls_layout.addWidget(k_group)

        controls_box.setLayout(controls_layout)
        center_layout.addWidget(controls_box)

        self.center_plot = PlotCanvas(self, width=3, height=3)
        center_layout.addWidget(QLabel("Selected Cache Rotation\nProjection"))
        center_layout.addWidget(self.center_plot)

        main_layout.addWidget(center_panel)

        # =========================================
        # ==== RIGHT PANEL (Closest Rotations) ====
        # =========================================
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        right_panel = QWidget()
        right_panel.setMaximumWidth(240)
        right_panel.setLayout(right_layout)

        title_layout = QHBoxLayout()
        title = QLabel("Closest Rotations in the Cache")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        title_layout.addWidget(title)

        self.closest_rotations_switch = QCheckBox("On")
        self.closest_rotations_switch.setChecked(True)
        self.closest_rotations_switch.setFont(QFont("Arial", 10))
        self.closest_rotations_switch.stateChanged.connect(self.toggle_closest_rotations)
        title_layout.addWidget(self.closest_rotations_switch)

        title_layout.addStretch()
        right_layout.addLayout(title_layout)

        self.cache_plots = []
        self.cache_checkboxes = []

        for i in range(3):
            plot_container = QWidget()
            plot_layout = QHBoxLayout(plot_container)
            plot_layout.setSpacing(5)
            plot_layout.setContentsMargins(0, 0, 0, 0)

            cache_plot = PlotCanvas(self, width=2.5, height=2.3)
            cache_plot.setMaximumHeight(170)
            cache_plot.ax.text(0.5, 0.5, 'Select a simulation\nrotation to see\nclosest matches',
                               ha='center', va='center',
                               transform=cache_plot.ax.transAxes, fontsize=8)
            cache_plot.ax.set_title(f"Closest #{i + 1}", fontsize=8)
            cache_plot.draw()
            self.cache_plots.append(cache_plot)
            plot_layout.addWidget(cache_plot)

            checkbox_container = QWidget()
            checkbox_layout = QVBoxLayout(checkbox_container)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_layout.setSpacing(2)
            checkbox_layout.setAlignment(Qt.AlignVCenter)

            checkboxes = {}

            t_checkbox = QCheckBox("T")
            t_checkbox.setEnabled(False)
            t_checkbox.setFont(QFont("Arial", 8))
            checkboxes['T'] = t_checkbox
            checkbox_layout.addWidget(t_checkbox)

            j_checkbox = QCheckBox("J")
            j_checkbox.setEnabled(False)
            j_checkbox.setFont(QFont("Arial", 8))
            checkboxes['J'] = j_checkbox
            checkbox_layout.addWidget(j_checkbox)

            if self.sym == "D2":
                coset_checkbox = QCheckBox("c")
                coset_checkbox.setEnabled(False)
                coset_checkbox.setFont(QFont("Arial", 8))
                coset_checkbox.setToolTip("O/D₂ coset representative")
                checkboxes['coset'] = coset_checkbox
                checkbox_layout.addWidget(coset_checkbox)
            elif self.is_cn:
                pi_x_checkbox = QCheckBox("π_x")
                pi_x_checkbox.setEnabled(False)
                pi_x_checkbox.setFont(QFont("Arial", 8))
                pi_x_checkbox.setToolTip("R_x(π) applied from left")
                checkboxes['pi_x'] = pi_x_checkbox
                checkbox_layout.addWidget(pi_x_checkbox)

                k_checkbox = QCheckBox("k=0")
                k_checkbox.setEnabled(False)
                k_checkbox.setFont(QFont("Arial", 8))
                k_checkbox.setToolTip("R_z(2πk/(nK)) applied from right")
                checkboxes['k'] = k_checkbox
                checkbox_layout.addWidget(k_checkbox)
            else:
                r_checkbox = QCheckBox("r")
                r_checkbox.setEnabled(False)
                r_checkbox.setFont(QFont("Arial", 8))
                checkboxes['r'] = r_checkbox
                checkbox_layout.addWidget(r_checkbox)

                if not (self.sym == "T" or (isinstance(self.sym, str) and self.sym.startswith("D"))):
                    r_checkbox.setVisible(False)

            self.cache_checkboxes.append(checkboxes)
            plot_layout.addWidget(checkbox_container)

            right_layout.addWidget(plot_container)

        main_layout.addWidget(right_panel)

        # =========================================
        # ==== HISTOGRAM PANEL (Far Right) ========
        # =========================================
        histogram_layout = QVBoxLayout()
        histogram_layout.setSpacing(10)
        histogram_panel = QWidget()
        histogram_panel.setLayout(histogram_layout)

        histogram_title = QLabel("Angular Differences")
        histogram_title.setFont(QFont("Arial", 12, QFont.Bold))
        histogram_title.setAlignment(Qt.AlignCenter)
        histogram_layout.addWidget(histogram_title)

        self.histogram_plot = PlotCanvas(self, width=3, height=4)
        self.histogram_plot.setMaximumHeight(350)
        histogram_layout.addWidget(self.histogram_plot)

        indices_title = QLabel("Angular Difference Indices")
        indices_title.setFont(QFont("Arial", 10, QFont.Bold))
        indices_title.setAlignment(Qt.AlignCenter)
        histogram_layout.addWidget(indices_title)

        less_than_10_group = QGroupBox("Angular Difference < 10°")
        less_than_10_layout = QVBoxLayout()
        self.less_than_10_list = QLabel("No Data")
        self.less_than_10_list.setWordWrap(True)
        less_than_10_layout.addWidget(self.less_than_10_list)
        less_than_10_group.setLayout(less_than_10_layout)
        histogram_layout.addWidget(less_than_10_group)

        more_than_10_group = QGroupBox("Angular Difference ≥ 10°")
        more_than_10_layout = QVBoxLayout()
        self.more_than_10_list = QLabel("No Data")
        self.more_than_10_list.setWordWrap(True)
        more_than_10_layout.addWidget(self.more_than_10_list)
        more_than_10_group.setLayout(more_than_10_layout)
        histogram_layout.addWidget(more_than_10_group)

        main_layout.addWidget(histogram_panel, 1)
        histogram_layout.addStretch()

    def on_cache_coset_view_changed(self, index):
        """Handle coset representative selection change for cache rotation VISUALIZATION"""
        self.current_cache_coset_index = index
        if self.cache_rotations is not None and hasattr(self, 'current_rotation_index'):
            self.update_selected_rotation_by_index(self.current_rotation_index,
                                                   self.get_selection_count(
                                                       self.current_sim_rotation_index) if self.current_sim_rotation_selected else None)

    def on_pi_x_view_changed(self, state):
        """Handle π_x checkbox change for cache rotation VISUALIZATION (C_n only)"""
        self.current_cache_pi_x_applied = (state == Qt.Checked)
        if self.cache_rotations is not None and hasattr(self, 'current_rotation_index'):
            self.update_selected_rotation_by_index(self.current_rotation_index,
                                                   self.get_selection_count(
                                                       self.current_sim_rotation_index) if self.current_sim_rotation_selected else None)

    # NEW: Callback for applying a global coset multiplier
    def apply_global_coset(self):
        """
        D2 Only: Sets a global coset multiplier for calculations and visualization.
        This does NOT modify the 'cache_coset_indices' data array.
        """
        self.global_coset_multiplier_index = self.global_coset_combo.currentIndex()
        target_name = self.coset_names_short[self.global_coset_multiplier_index]

        self.progress_label.setText(f"Applying global coset: {target_name}")

        # Recalculate histogram and stats with the new global multiplier
        self.update_angular_histogram()

        # Refresh current cache view to reflect the new global multiplier
        self.update_selected_rotation()

    def apply_global_pi_x(self):
        """
        C_n Only: Sets a global π_x multiplier for calculations and visualization.
        This does NOT modify the 'cache_pi_x_indices' data array.
        """
        self.global_pi_x_applied = self.global_pi_x_checkbox.isChecked()

        status = "ON" if self.global_pi_x_applied else "OFF"
        self.progress_label.setText(f"Global R_x(π): {status}")

        # Recalculate histogram and stats with the new global multiplier
        self.update_angular_histogram()

        # Refresh current cache view to reflect the new global multiplier
        self.update_selected_rotation()

    def apply_k_value(self):
        """
        C_n Only: Apply a new K value, recalculate best rotations and update histogram.
        """
        try:
            new_k = int(self.k_value_input.text())
            if new_k < 1:
                self.progress_label.setText("K must be >= 1")
                return

            old_k = self.K_value
            self.K_value = new_k

            # Update the visual and global k combo boxes with new range
            num_k_values = self.cn_order * self.K_value
            self.visual_k_combo.blockSignals(True)
            self.global_k_combo.blockSignals(True)

            self.visual_k_combo.clear()
            self.global_k_combo.clear()
            self.visual_k_combo.addItems([str(i) for i in range(num_k_values)])
            self.global_k_combo.addItems([str(i) for i in range(num_k_values)])

            # Reset k indices to 0 when K changes
            self.current_cache_k_index = 0
            self.global_k_index = 0
            self.visual_k_combo.setCurrentIndex(0)
            self.global_k_combo.setCurrentIndex(0)

            self.visual_k_combo.blockSignals(False)
            self.global_k_combo.blockSignals(False)

            # Reset k indices in data if they exist
            if (self.cache_selected_data is not None and
                    'cache_k_indices' in self.cache_selected_data):
                self.cache_selected_data['cache_k_indices'] = np.zeros(
                    len(self.cache_selected_data['cache_k_indices']), dtype=int)

            self.progress_label.setText(f"Applied K={new_k}. Recalculating...")

            # Recalculate histogram with new K
            self.update_angular_histogram()

            # Refresh current views
            if self.current_sim_rotation_selected:
                self.update_simulation_rotation_display()
            self.update_selected_rotation()

        except ValueError:
            self.progress_label.setText("Please enter a valid integer for K")

    def on_visual_k_changed(self, index):
        """Handle visual k selection change for cache rotation VISUALIZATION"""
        self.current_cache_k_index = index
        if self.cache_rotations is not None and hasattr(self, 'current_rotation_index'):
            self.update_selected_rotation_by_index(self.current_rotation_index,
                                                   self.get_selection_count(
                                                       self.current_sim_rotation_index) if self.current_sim_rotation_selected else None)

    def apply_global_k(self):
        """
        C_n Only: Apply a global k index for all rotations in histogram calculation.
        """
        self.global_k_index = self.global_k_combo.currentIndex()

        self.progress_label.setText(f"Applied global k={self.global_k_index}")

        # Recalculate histogram
        self.update_angular_histogram()

        # Refresh current view
        self.update_selected_rotation()

    def get_transformed_sim_rotation(self, index):
        """Get simulation rotation. LOCKED: No longer accepts UI modifications."""
        if self.sim_rotations is None or index >= len(self.sim_rotations):
            return None

        # Return the raw, unmodified simulation rotation
        return self.sim_rotations[index].copy()

    def flip_j_values(self):
        """Flip all J values in cache_selected_data by multiplying by -1"""
        if (self.cache_selected_data is not None and
                'cache_J_indices' in self.cache_selected_data):
            self.cache_selected_data['cache_J_indices'] *= -1
            self.progress_label.setText("Flipped all J values")
            self.update_angular_histogram()
        else:
            self.progress_label.setText("No J values to flip")

    def flip_r_values(self):
        """Flip all r values in cache_selected_data by multiplying by -1"""
        if (self.cache_selected_data is not None and
                'cache_selected_r_indices' in self.cache_selected_data):
            self.cache_selected_data['cache_selected_r_indices'] *= -1
            self.progress_label.setText("Flipped all r values")
            self.update_angular_histogram()
        else:
            self.progress_label.setText("No r values to flip")

    def flip_pi_x_values(self):
        """Flip all π_x values in cache_selected_data by multiplying by -1 (C_n only)"""
        if (self.cache_selected_data is not None and
                'cache_pi_x_indices' in self.cache_selected_data):
            self.cache_selected_data['cache_pi_x_indices'] *= -1
            self.progress_label.setText("Flipped all π_x values")
            self.update_angular_histogram()
            # Also refresh the current view
            if self.current_sim_rotation_selected:
                self.update_pi_x_checkbox(self.current_sim_rotation_index)
                self.update_selected_rotation()
        else:
            self.progress_label.setText("No π_x values to flip")

    def load_data(self):
        """Load and process data in a separate thread"""
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("Loading data...")

        self.data_processor = DataProcessor(
            sym=self.sym,
            cache_file=self.cache_file,
            simulation_rotations=self.simulation_rotations,
            cache_selected_data=self.cache_selected_data
        )
        self.data_processor.data_ready.connect(self.on_data_ready)
        self.data_processor.progress_update.connect(self.on_progress_update)
        self.data_processor.start()

    def on_progress_update(self, message):
        """Update progress display"""
        self.progress_label.setText(message)

    def on_data_ready(self, volume, cache_rotations, cache_data, sim_rotations, cache_selected_data):
        """Handle processed data from the worker thread"""
        self.volume = volume
        self.cache_rotations = cache_rotations
        self.cache_data = cache_data
        self.sim_rotations = sim_rotations
        self.cache_selected_data = cache_selected_data

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_label.setText("Data loaded successfully!")

        if self.cache_rotations is not None:
            n_rotations = len(self.cache_rotations)
            self.rotation_count_label.setText(f"Available cache rotations: {n_rotations}")

        if self.sim_rotations is not None:
            n_sim_rotations = len(self.sim_rotations)
            self.sim_rotation_count_label.setText(f"Total simulation rotations:\n{n_sim_rotations}")
        else:
            self.sim_rotation_count_label.setText("No simulation rotations\nprovided")

        if (self.cache_selected_data is not None and
                'cache_selected_inds' in self.cache_selected_data and
                self.sim_rotations is not None):
            cache_selected_inds = self.cache_selected_data['cache_selected_inds']
            if len(cache_selected_inds) != len(self.sim_rotations):
                self.progress_label.setText(
                    f"Warning: cache_selected_inds length ({len(cache_selected_inds)}) doesn't match simulation rotations length ({len(self.sim_rotations)})")

        # Show/hide r controls based on symmetry (hide for D2 and C_n, show for T and other D symmetries)
        if self.sym == "T" or (isinstance(self.sym, str) and self.sym.startswith("D") and self.sym != "D2"):
            self.r_value_checkbox.setVisible(True)
            self.flip_r_button.setVisible(True)
            for checkboxes in self.cache_checkboxes:
                if 'r' in checkboxes:
                    checkboxes['r'].setVisible(True)
        else:
            self.r_value_checkbox.setVisible(False)
            self.flip_r_button.setVisible(False)
            for checkboxes in self.cache_checkboxes:
                if 'r' in checkboxes:
                    checkboxes['r'].setVisible(False)

        self.show_empty_plots()
        self.update_angular_histogram()

    def show_empty_plots(self):
        """Show empty plots initially"""
        if self.sim_rotations is not None:
            self.left_plot.ax.clear()
            self.left_plot.ax.text(0.5, 0.5, 'Select a simulation rotation\nindex to view projection',
                                   ha='center', va='center', transform=self.left_plot.ax.transAxes)
            self.left_plot.ax.set_title("Simulation Rotation\nProjection", fontsize=10)
            self.left_plot.draw()
        else:
            self.left_plot.ax.clear()
            self.left_plot.ax.text(0.5, 0.5, 'No simulation rotations\nprovided',
                                   ha='center', va='center', transform=self.left_plot.ax.transAxes)
            self.left_plot.ax.set_title("Simulation Rotations", fontsize=10)
            self.left_plot.draw()

        self.center_plot.ax.clear()
        self.center_plot.ax.text(0.5, 0.5, 'Select a cache rotation index\nto view projection',
                                 ha='center', va='center', transform=self.center_plot.ax.transAxes)
        self.center_plot.ax.set_title("Selected Cache Rotation\nProjection", fontsize=10)
        self.center_plot.draw()

    def update_j_value_checkbox(self, simulation_index):
        """Update the J value checkbox based on the current data arrays"""
        if (self.cache_selected_data is not None and
                'cache_J_indices' in self.cache_selected_data and
                'cache_selected_inds' in self.cache_selected_data and
                self.current_sim_rotation_selected):
            cache_J_indices = self.cache_selected_data['cache_J_indices']
            j_value = cache_J_indices[simulation_index]
            self.j_value_checkbox.setChecked(bool(j_value == -1))

    def update_r_value_checkbox(self, simulation_index):
        """Update the R value checkbox based on the current data arrays"""
        if (self.cache_selected_data is not None and
                'cache_selected_r_indices' in self.cache_selected_data and
                'cache_selected_inds' in self.cache_selected_data and
                self.current_sim_rotation_selected and
                self.sym != "D2" and
                not self.is_cn and
                (self.sym == "T" or (isinstance(self.sym, str) and self.sym.startswith("D")))):
            cache_r_indices = self.cache_selected_data['cache_selected_r_indices']
            r_value = cache_r_indices[simulation_index]
            self.r_value_checkbox.setChecked(bool(r_value == -1))

    def update_coset_value_combo(self, simulation_index):
        """Update the VISUAL coset combo box based on the current data arrays"""
        if (self.sym == "D2" and
                self.cache_selected_data is not None and
                'cache_coset_indices' in self.cache_selected_data and
                'cache_selected_inds' in self.cache_selected_data and
                self.current_sim_rotation_selected and
                hasattr(self, 'cache_coset_combo')):
            cache_coset_indices = self.cache_selected_data['cache_coset_indices']
            coset_value = cache_coset_indices[simulation_index]

            # Sync the VISUAL dropdown to the DATA value
            self.cache_coset_combo.setCurrentIndex(int(coset_value))

    def update_pi_x_checkbox(self, simulation_index):
        """Update the VISUAL π_x checkbox based on the current data arrays (C_n only)"""
        if (self.is_cn and
                self.cache_selected_data is not None and
                'cache_pi_x_indices' in self.cache_selected_data and
                'cache_selected_inds' in self.cache_selected_data and
                self.current_sim_rotation_selected and
                hasattr(self, 'pi_x_checkbox')):
            cache_pi_x_indices = self.cache_selected_data['cache_pi_x_indices']
            pi_x_value = cache_pi_x_indices[simulation_index]

            # Sync the VISUAL checkbox to the DATA value
            self.pi_x_checkbox.setChecked(bool(pi_x_value == -1))

    def update_k_combo(self, simulation_index):
        """Update the VISUAL k combo box based on the current data arrays (C_n only)"""
        if (self.is_cn and
                self.cache_selected_data is not None and
                'cache_k_indices' in self.cache_selected_data and
                'cache_selected_inds' in self.cache_selected_data and
                self.current_sim_rotation_selected and
                hasattr(self, 'visual_k_combo')):
            cache_k_indices = self.cache_selected_data['cache_k_indices']
            k_value = cache_k_indices[simulation_index]

            # Sync the VISUAL combo to the DATA value (within valid range)
            max_k = self.cn_order * self.K_value
            if 0 <= k_value < max_k:
                self.visual_k_combo.setCurrentIndex(int(k_value))
            else:
                self.visual_k_combo.setCurrentIndex(0)

    def get_selection_count(self, simulation_index):
        """Get the selection count for a cache index if available"""
        if (self.cache_selected_data is not None and
                'cache_selected_count' in self.cache_selected_data and
                'cache_selected_inds' in self.cache_selected_data):
            cache_selected_count = self.cache_selected_data['cache_selected_count']
            return cache_selected_count[simulation_index]
        return None

    def toggle_closest_rotations(self):
        """Toggle whether closest rotations are computed and displayed"""
        if self.closest_rotations_switch.isChecked():
            self.closest_rotations_switch.setText("On")
            if (self.current_sim_rotation_selected and
                    self.sim_rotations is not None and
                    self.current_sim_rotation_index < len(self.sim_rotations)):
                # Use the raw sim rotation for the search
                rotation_matrix = self.get_transformed_sim_rotation(self.current_sim_rotation_index)
                self.update_closest_cache_rotations(rotation_matrix)
        else:
            self.closest_rotations_switch.setText("Off")
            self.clear_closest_rotation_plots()

    def clear_closest_rotation_plots(self):
        """Clear all closest rotation plots and checkboxes"""
        for i in range(len(self.cache_plots)):
            self.cache_plots[i].ax.clear()
            self.cache_plots[i].ax.text(0.5, 0.5, 'Closest rotations\ncomputation is off',
                                        ha='center', va='center',
                                        transform=self.cache_plots[i].ax.transAxes, fontsize=8)
            self.cache_plots[i].ax.set_title(f"Closest #{i + 1}", fontsize=8)
            self.cache_plots[i].draw()

            self.cache_checkboxes[i]['T'].setChecked(False)
            self.cache_checkboxes[i]['J'].setChecked(False)
            if self.sym == "D2" and 'coset' in self.cache_checkboxes[i]:
                self.cache_checkboxes[i]['coset'].setChecked(False)
            elif self.is_cn and 'pi_x' in self.cache_checkboxes[i]:
                self.cache_checkboxes[i]['pi_x'].setChecked(False)
            elif 'r' in self.cache_checkboxes[i]:
                self.cache_checkboxes[i]['r'].setChecked(False)

    def update_closest_cache_rotations(self, sim_rotation_matrix):
        """Find and display the closest cache rotations to the simulation rotation"""
        if self.cache_rotations is None or self.volume is None or sim_rotation_matrix is None:
            return

        if not self.closest_rotations_switch.isChecked():
            return

        self.progress_label.setText("Finding closest cache rotations...")

        closest_rotations = self.find_closest_cache_rotations(sim_rotation_matrix, n_closest=3)
        self.closest_cache_indices = [idx for idx, _, _, _, _, _, _, _ in closest_rotations]

        for i, rotation_data in enumerate(closest_rotations):
            if i < len(self.cache_plots):
                cache_idx, ang_dist, t_val, j_val, r_val, coset_idx, pi_x_val, k_val = rotation_data

                try:
                    cache_rotation = self.cache_rotations[cache_idx]
                    transformed_matrix = cache_rotation.copy()

                    # Apply D2 coset representative (replaces r functionality for D2)
                    if self.sym == "D2" and self.coset_reps is not None:
                        coset_rep = self.coset_reps[coset_idx]
                        transformed_matrix = coset_rep @ transformed_matrix

                    if t_val == -1:
                        transformed_matrix = transformed_matrix.T

                    # Apply π rotation about x-axis for C_n (from the left)
                    if self.is_cn and pi_x_val == -1:
                        transformed_matrix = self.pi_x_rotation @ transformed_matrix

                    # Apply k rotation about z-axis for C_n (from the right)
                    if self.is_cn and k_val != 0:
                        angle = 2 * np.pi * k_val / (self.cn_order * self.K_value)
                        transformed_matrix = transformed_matrix @ get_z_rotation(angle)

                    # Apply r for non-D2, non-C_n symmetries
                    if r_val == -1 and self.sym != "D2" and not self.is_cn and (
                            self.sym == "T" or (isinstance(self.sym, str) and self.sym.startswith("D"))):
                        normalizer_group = group_elements(normalizer(self.sym))
                        r = normalizer_group[-1]
                        transformed_matrix = r @ transformed_matrix

                    if j_val == -1:
                        transformed_matrix = multi_Jify(transformed_matrix)

                    rot_for_projection = transformed_matrix.reshape(1, 3, 3).astype(np.float32)
                    projection = self.volume.project(rot_for_projection)

                    if hasattr(projection, 'asnumpy'):
                        proj_array = projection.asnumpy()
                        proj_image = proj_array[0] if len(proj_array.shape) >= 3 else proj_array
                    else:
                        proj_image = projection

                    # Add coset name to title if D2
                    title = f"Cache #{cache_idx}"
                    if self.sym == "D2":
                        title += f" ({self.coset_names_short[coset_idx]})"
                    elif self.is_cn:
                        extras = []
                        if pi_x_val == -1:
                            extras.append("π_x")
                        if k_val != 0:
                            extras.append(f"k={k_val}")
                        if extras:
                            title += f" ({', '.join(extras)})"

                    self.cache_plots[i].plot_image(
                        proj_image,
                        title,
                        f"Angle: {ang_dist:.2f}°"
                    )

                    # Update checkboxes
                    self.cache_checkboxes[i]['T'].setChecked(bool(t_val == -1))
                    self.cache_checkboxes[i]['J'].setChecked(bool(j_val == -1))

                    if self.sym == "D2" and 'coset' in self.cache_checkboxes[i]:
                        self.cache_checkboxes[i]['coset'].setChecked(bool(coset_idx != 0))
                        self.cache_checkboxes[i]['coset'].setText(f"c:{self.coset_names_short[coset_idx]}")
                    elif self.is_cn and 'pi_x' in self.cache_checkboxes[i]:
                        self.cache_checkboxes[i]['pi_x'].setChecked(bool(pi_x_val == -1))
                        # Update k label
                        if 'k' in self.cache_checkboxes[i]:
                            self.cache_checkboxes[i]['k'].setText(f"k={k_val}")
                            self.cache_checkboxes[i]['k'].setChecked(bool(k_val != 0))
                    elif 'r' in self.cache_checkboxes[i]:
                        self.cache_checkboxes[i]['r'].setChecked(bool(r_val == -1))

                    # Update visibility
                    if self.sym == "T" or (isinstance(self.sym, str) and self.sym.startswith("D") and self.sym != "D2"):
                        if 'r' in self.cache_checkboxes[i]:
                            self.cache_checkboxes[i]['r'].setVisible(True)
                    else:
                        if 'r' in self.cache_checkboxes[i]:
                            self.cache_checkboxes[i]['r'].setVisible(False)

                except Exception as e:
                    self.cache_plots[i].ax.clear()
                    self.cache_plots[i].ax.text(0.5, 0.5, f'Error: {str(e)}',
                                                ha='center', va='center',
                                                transform=self.cache_plots[i].ax.transAxes,
                                                fontsize=8)
                    self.cache_plots[i].draw()
                    self.cache_checkboxes[i]['T'].setChecked(False)
                    self.cache_checkboxes[i]['J'].setChecked(False)
                    if self.sym == "D2" and 'coset' in self.cache_checkboxes[i]:
                        self.cache_checkboxes[i]['coset'].setChecked(False)
                    elif self.is_cn and 'pi_x' in self.cache_checkboxes[i]:
                        self.cache_checkboxes[i]['pi_x'].setChecked(False)
                    elif 'r' in self.cache_checkboxes[i]:
                        self.cache_checkboxes[i]['r'].setChecked(False)

        for i in range(len(closest_rotations), len(self.cache_plots)):
            self.cache_plots[i].ax.clear()
            self.cache_plots[i].ax.text(0.5, 0.5, 'No match',
                                        ha='center', va='center',
                                        transform=self.cache_plots[i].ax.transAxes,
                                        fontsize=8)
            self.cache_plots[i].draw()
            self.cache_checkboxes[i]['T'].setChecked(False)
            self.cache_checkboxes[i]['J'].setChecked(False)
            if self.sym == "D2" and 'coset' in self.cache_checkboxes[i]:
                self.cache_checkboxes[i]['coset'].setChecked(False)
            elif self.is_cn and 'pi_x' in self.cache_checkboxes[i]:
                self.cache_checkboxes[i]['pi_x'].setChecked(False)
            elif 'r' in self.cache_checkboxes[i]:
                self.cache_checkboxes[i]['r'].setChecked(False)

        self.progress_label.setText(f"Found {len(closest_rotations)} closest cache rotations")

    def update_simulation_rotation(self):
        """Update the simulation rotation display"""
        try:
            if self.sim_rotations is None:
                self.progress_label.setText("No simulation rotations available")
                return

            index = int(self.sim_index_input.text())
            n_sim_rotations = len(self.sim_rotations)

            if 0 <= index < n_sim_rotations:
                self.current_sim_rotation_index = index
                self.current_sim_rotation_selected = True

                # Sync center panel UI elements to the data for this index
                self.update_j_value_checkbox(index)
                if self.sym != "D2" and not self.is_cn:
                    self.update_r_value_checkbox(index)
                elif self.sym == "D2":
                    self.update_coset_value_combo(index)  # This syncs the *visual* dropdown
                elif self.is_cn:
                    self.update_pi_x_checkbox(index)  # This syncs the *visual* checkbox
                    self.update_k_combo(index)  # This syncs the *visual* k combo

                selection_count = self.get_selection_count(index)

                # Update the left (simulation) plot
                self.update_simulation_rotation_display()

                if (self.cache_selected_data is not None and
                        'cache_selected_inds' in self.cache_selected_data and
                        index < len(self.cache_selected_data['cache_selected_inds'])):

                    corresponding_cache_index = self.cache_selected_data['cache_selected_inds'][index]
                    self.index_input.setText(str(corresponding_cache_index))
                    self.update_selected_rotation_by_index(corresponding_cache_index, selection_count)
                    self.progress_label.setText(
                        f"Showing simulation rotation {index} and corresponding cache rotation {corresponding_cache_index}")
                else:
                    self.progress_label.setText(f"Showing simulation rotation {index}")

            else:
                self.progress_label.setText(f"Index {index} out of range (0-{n_sim_rotations - 1})")

        except ValueError:
            self.progress_label.setText("Please enter a valid integer for simulation rotation")

    def update_simulation_rotation_display(self):
        """Update the display of the current simulation rotation (Ground Truth)"""
        index = self.current_sim_rotation_index
        rotation_matrix = self.get_transformed_sim_rotation(index)

        if rotation_matrix is None:
            return

        self.progress_label.setText(f"Computing projection for simulation rotation {index}...")

        try:
            if hasattr(rotation_matrix, 'shape') and rotation_matrix.shape == (3, 3):
                rot_for_projection = rotation_matrix.reshape(1, 3, 3).astype(np.float32)
                projection = self.volume.project(rot_for_projection)

                if hasattr(projection, 'asnumpy'):
                    proj_array = projection.asnumpy()
                    proj_image = proj_array[0] if len(proj_array.shape) >= 3 else proj_array
                else:
                    proj_image = projection

                title = f"Simulation Rotation #{index}\n(Ground Truth)"
                self.left_plot.plot_image(proj_image, title)

                self.update_closest_cache_rotations(rotation_matrix)

            else:
                self.progress_label.setText(f"Invalid rotation matrix shape: {rotation_matrix.shape}")

        except Exception as e:
            self.progress_label.setText(f"Error computing projection: {str(e)}")
            dummy_proj = np.random.rand(64, 64)
            self.left_plot.plot_image(dummy_proj, f"Dummy Sim Projection {index}")

    def update_selected_rotation(self):
        """Update the cache rotation display (called from user input)"""
        try:
            index = int(self.index_input.text())
            self.update_selected_rotation_by_index(index)
        except ValueError:
            self.progress_label.setText("Please enter a valid integer for cache rotation")

    def update_selected_rotation_by_index(self, index, selection_count=None):
        """Update the cache rotation display by index"""
        if self.cache_rotations is not None and self.volume is not None:
            n_rotations = len(self.cache_rotations)

            if 0 <= index < n_rotations:
                self.current_rotation_index = index
                rotation_matrix = self.cache_rotations[index]
                self.progress_label.setText(f"Computing projection for cache rotation {index}...")

                try:
                    if hasattr(rotation_matrix, 'shape') and rotation_matrix.shape == (3, 3):
                        display_matrix = rotation_matrix.copy()

                        # Apply D2 coset representative from VISUAL dropdown
                        if self.sym == "D2" and self.coset_reps is not None:
                            # 1. Apply VISUAL coset rep from dropdown
                            visual_coset_rep = self.coset_reps[self.current_cache_coset_index]
                            display_matrix = visual_coset_rep @ display_matrix

                            # 2. Apply GLOBAL coset rep
                            global_coset_rep = self.coset_reps[self.global_coset_multiplier_index]
                            display_matrix = global_coset_rep @ display_matrix

                        # Apply π rotation about x-axis for C_n (from the left)
                        if self.is_cn:
                            # 1. Apply VISUAL pi_x from checkbox
                            if hasattr(self, 'pi_x_checkbox') and self.pi_x_checkbox.isChecked():
                                display_matrix = self.pi_x_rotation @ display_matrix

                            # 2. Apply GLOBAL pi_x
                            if self.global_pi_x_applied:
                                display_matrix = self.pi_x_rotation @ display_matrix

                            # 3. Apply VISUAL k rotation from combo (from the right)
                            if hasattr(self, 'visual_k_combo') and self.current_cache_k_index != 0:
                                angle = 2 * np.pi * self.current_cache_k_index / (self.cn_order * self.K_value)
                                display_matrix = display_matrix @ get_z_rotation(angle)

                            # 4. Apply GLOBAL k rotation (from the right)
                            if self.global_k_index != 0:
                                angle = 2 * np.pi * self.global_k_index / (self.cn_order * self.K_value)
                                display_matrix = display_matrix @ get_z_rotation(angle)

                        # Apply r transformation for non-D2, non-C_n symmetries from checkbox
                        if (self.r_value_checkbox.isVisible() and
                                self.r_value_checkbox.isChecked() and
                                self.sym != "D2" and
                                not self.is_cn):
                            normalizer_group = group_elements(normalizer(self.sym))
                            r = normalizer_group[-1]
                            display_matrix = r @ display_matrix

                        # Apply J from checkbox
                        if self.j_value_checkbox.isChecked():
                            display_matrix = multi_Jify(display_matrix)

                        rot_for_projection = display_matrix.reshape(1, 3, 3).astype(np.float32)
                        projection = self.volume.project(rot_for_projection)

                        if hasattr(projection, 'asnumpy'):
                            proj_array = projection.asnumpy()
                            proj_image = proj_array[0] if len(proj_array.shape) >= 3 else proj_array
                        else:
                            proj_image = projection

                        bottom_text = ""
                        count_text = ""

                        if (self.sim_rotations is not None and
                                self.current_sim_rotation_selected and
                                hasattr(self, 'current_sim_rotation_index') and
                                self.current_sim_rotation_index < len(self.sim_rotations)):

                            sim_rotation = self.get_transformed_sim_rotation(self.current_sim_rotation_index)
                            mse = self.calculate_matrix_mse(display_matrix, sim_rotation)
                            ang_dist = self.calculate_angular_distance(display_matrix, sim_rotation)

                            try:
                                if mse is not None:
                                    bottom_text = f"MSE: {mse:.6f}\nAngular Distance: {ang_dist:.2f}°"
                                else:
                                    bottom_text = f"Angular Distance: {ang_dist:.2f}°"
                            except Exception as e:
                                bottom_text = "Angular Distance: Error"

                        if selection_count is not None and self.current_sim_rotation_selected:
                            count_text = f"Selection count: {selection_count}"

                        # Update title with VISUAL and GLOBAL info
                        title = f"Cache Rotation #{index} Projection"
                        if self.sym == "D2":
                            view_str = self.coset_names_short[self.current_cache_coset_index]
                            global_str = self.coset_names_short[self.global_coset_multiplier_index]

                            title_parts = [f"Cache #{index}"]
                            if self.current_cache_coset_index != 0:
                                title_parts.append(f"View: {view_str}")
                            if self.global_coset_multiplier_index != 0:
                                title_parts.append(f"Global: {global_str}")

                            title = "\n".join(title_parts)
                        elif self.is_cn:
                            title_parts = [f"Cache #{index}"]
                            view_parts = []
                            global_parts = []
                            if hasattr(self, 'pi_x_checkbox') and self.pi_x_checkbox.isChecked():
                                view_parts.append("π_x")
                            if self.current_cache_k_index != 0:
                                view_parts.append(f"k={self.current_cache_k_index}")
                            if self.global_pi_x_applied:
                                global_parts.append("π_x")
                            if self.global_k_index != 0:
                                global_parts.append(f"k={self.global_k_index}")

                            if view_parts:
                                title_parts.append(f"View: {', '.join(view_parts)}")
                            if global_parts:
                                title_parts.append(f"Global: {', '.join(global_parts)}")

                            title = "\n".join(title_parts)

                        self.center_plot.plot_image(proj_image, title, bottom_text, count_text)

                    else:
                        self.progress_label.setText(f"Invalid rotation matrix shape: {rotation_matrix.shape}")

                except Exception as e:
                    self.progress_label.setText(f"Error computing projection: {str(e)}")
                    dummy_proj = np.random.rand(64, 64)
                    self.center_plot.plot_image(dummy_proj, f"Dummy Cache Projection {index}")

            else:
                self.progress_label.setText(f"Index {index} out of range (0-{n_rotations - 1})")
        else:
            self.progress_label.setText("Data not loaded yet")


def create_projection_viewer_simulation(sym='T', cache_file=None, simulation_rotations=None, cache_selected_data=None):
    """
    Factory function to create and show the rotation viewer.

    Args:
        sym (str): Symmetry type ('T', 'O', 'I', 'D2', 'C3', 'C4', etc.)
        cache_file (str): Path to the cache file containing rotation matrices
        simulation_rotations (np.ndarray): Array of simulation rotation matrices
        cache_selected_data (dict): Dictionary containing:
            - 'cache_selected_inds': Array of cache indices corresponding to each simulation rotation
            - 'cache_J_indices': Array of J values (-1 or 1) for each simulation rotation
            - 'cache_selected_count': Array of selection counts for each rotation
            - 'cache_selected_r_indices': Array of r values (-1 or 1) for normalizer coset (T/D symmetries except D2)
            - 'cache_coset_indices': Array of coset indices (0-5) for O/D2 coset representatives (D2 only)
            - 'cache_pi_x_indices': Array of π_x values (-1 or 1) for C_n symmetries (left multiply by R_x(π))
            - 'cache_k_indices': Array of k indices for C_n symmetries (right multiply by R_z(2πk/(n*K)))

    Returns:
        ProjectionViewer: The created viewer widget
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = ProjectionViewer(sym=sym, cache_file=cache_file, simulation_rotations=simulation_rotations,
                              cache_selected_data=cache_selected_data)
    viewer.show()
    app.exec_()
    return viewer


if __name__ == "__main__":
    # Example usage with C_n symmetry:
    sym = 'C4'  # Can be 'C2', 'C3', 'C4', 'C5', etc.

    # Create a dummy cache file if it doesn't exist for testing
    cache_file = ROTATIONS_CACHE_DIR / f"cache_{sym}_debug_example.pkl"
    if not cache_file.exists():
        print("Creating dummy cache file...")
        dummy_cache_rotations = np.array([np.eye(3) for _ in range(150)])  # 150 dummy rotations
        with open(cache_file, 'wb') as f:
            pickle.dump([dummy_cache_rotations], f)

    sim_size = 400

    # Create dummy sim rotations (e.g., noisy identity matrices)
    simulation_rotations = np.array([np.eye(3) + np.random.rand(3, 3) * 0.1 for _ in range(sim_size)])

    # For C_n, use cache_pi_x_indices and cache_k_indices
    cache_selected_data = {
        "cache_selected_inds": np.random.randint(0, 150, sim_size),  # Indices into the cache
        "cache_J_indices": np.random.choice([-1, 1], sim_size),
        "cache_selected_count": np.random.randint(1, 10, sim_size),
        "cache_pi_x_indices": np.random.choice([-1, 1], sim_size),  # For C_n: -1 = apply R_x(π), 1 = no transform
        "cache_k_indices": np.zeros(sim_size, dtype=int),  # For C_n: k index for R_z(2πk/(n*K))
    }

    create_projection_viewer_simulation(sym=sym, cache_file=cache_file, simulation_rotations=simulation_rotations,
                                        cache_selected_data=cache_selected_data)