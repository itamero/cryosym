import sys

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QCheckBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
import logging
import pickle
import os

from cryosym.volume_download.data_downloader import data_downloader
from aspire.volume import Volume
from cryosym.group_elements import group_elements, normalizer
from cryosym.utils import multi_Jify
from cryosym.config import ROTATIONS_CACHE_DIR

logger = logging.getLogger(__name__)


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
            self.ax.set_title(title, fontsize=10)
            self.ax.axis('off')

            # Add bottom text if provided
            if bottom_text:
                self.ax.text(0.5, -0.05, bottom_text, ha='center', va='top',
                             transform=self.ax.transAxes, fontsize=9, color='blue')

            # Add count text in green if provided
            if count_text:
                self.ax.text(0.5, -0.15, count_text, ha='center', va='top',
                             transform=self.ax.transAxes, fontsize=9, color='green')
        else:
            self.ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_title(title, fontsize=10)
        self.draw()


class DummyVolume:
    """Dummy volume class for testing"""

    def project(self, rotations):
        # Return dummy projections
        n_rotations = rotations.shape[0] if len(rotations.shape) > 2 else 1
        return np.random.rand(n_rotations, 64, 64)


class DataProcessor(QThread):
    """Thread for processing volume data without blocking the GUI"""
    data_ready = pyqtSignal(object, object, object, object, object)  # volume, cache_rotations, cache_data, simulation_projections, cache_selected_data
    progress_update = pyqtSignal(str)

    def __init__(self, sym='T', cache_file=None, simulation_projections=None, cache_selected_data=None, img_size=70,
                 num_imgs=12):
        super().__init__()
        self.sym = sym
        self.cache_file = cache_file
        self.simulation_projections = simulation_projections
        self.cache_selected_data = cache_selected_data
        self.img_size = img_size
        self.num_imgs = num_imgs

    def run(self):
        try:
            # Load the cache file
            self.progress_update.emit("Loading rotation cache...")
            cache_data = None
            R = None

            if self.cache_file and (isinstance(self.cache_file, str) and os.path.exists(self.cache_file) or (hasattr(self.cache_file, 'exists') and self.cache_file.exists())):
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    if isinstance(cache_data, (list, tuple)) and len(cache_data) >= 1:
                        R = cache_data[0]  # First argument is the rotation matrices
                        self.progress_update.emit(f"Loaded {len(R)} cached rotations")
                    else:
                        raise ValueError("Invalid cache file format")
            else:
                self.progress_update.emit("Cache file not found or not provided, creating dummy data...")
                # Create dummy rotation matrices for testing
                R = np.random.rand(100, 3, 3)
                cache_data = [R, None, None, None, None]

            self.progress_update.emit("Loading volume...")

            # Try to load volume using the existing system
            try:
                volume_file = data_downloader(self.sym)
                og_vol = Volume.load(volume_file)
                og_vol = og_vol.downsample(70)
            except:
                # Fallback to dummy volume
                self.progress_update.emit("Using dummy volume for testing...")
                og_vol = DummyVolume()

            # Process simulation projections
            processed_sim_projections = None
            if self.simulation_projections is not None:
                self.progress_update.emit("Processing simulation projections...")
                if isinstance(self.simulation_projections, np.ndarray):
                    processed_sim_projections = self.simulation_projections
                    self.progress_update.emit(f"Loaded {len(processed_sim_projections)} simulation projections")
                else:
                    # Create dummy simulation projections
                    processed_sim_projections = np.random.rand(50, 64, 64)
                    self.progress_update.emit("Created dummy simulation projections")

            # Process cache_selected_data
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
            self.data_ready.emit(og_vol, R, cache_data, processed_sim_projections, processed_cache_selected_data)

        except Exception as e:
            self.progress_update.emit(f"Error: {str(e)}")
            # Emit dummy data for testing
            dummy_rotations = np.random.rand(50, 3, 3)
            dummy_volume = DummyVolume()
            dummy_cache = [dummy_rotations, None, None, None, None]
            dummy_sim_projections = np.random.rand(25, 64, 64) if self.simulation_projections is not None else None
            # Create dummy cache_selected_data
            dummy_cache_selected_data = None
            if self.cache_selected_data is not None:
                dummy_cache_selected_data = {
                    "cache_selected_inds": np.random.randint(0, 50, 25),
                    "cache_J_indices": np.random.choice([-1, 1], 25),
                    "cache_selected_count": np.random.randint(1, 10, 25),
                    "cache_selected_r_indices": np.random.choice([-1, 1], 25)
                }
            self.data_ready.emit(dummy_volume, dummy_rotations, dummy_cache, dummy_sim_projections,
                                 dummy_cache_selected_data)


class ProjectionViewer(QWidget):
    def __init__(self, sym='T', cache_file=None, simulation_projections=None, cache_selected_data=None):
        super().__init__()
        self.sym = sym
        self.cache_file = cache_file
        self.simulation_projections = simulation_projections
        self.cache_selected_data = cache_selected_data

        self.setWindowTitle(f"Projection Viewer (Class Averages) - {sym} Symmetry")
        self.setMinimumSize(900, 700)

        # Data storage
        self.volume = None
        self.cache_rotations = None
        self.cache_data = None
        self.sim_projections = None
        self.current_rotation_index = 0
        self.current_sim_projection_index = 0
        self.current_sim_projection_selected = False

        self.init_ui()
        self.load_data()

    def init_ui(self):
        # Main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # ==== LEFT PANEL ====
        left_layout = QVBoxLayout()

        # Class Averages projections display
        sim_title = QLabel("Class Averages Projections")
        sim_title.setFont(QFont("Arial", 12, QFont.Bold))
        sim_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(sim_title)

        # Class Averages projection count and index input
        self.sim_projection_count_label = QLabel("Total class averages projections: Loading...")
        self.sim_projection_count_label.setFont(QFont("Arial", 10))
        self.sim_projection_count_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.sim_projection_count_label)

        # Input for simulation projection index
        sim_index_layout = QHBoxLayout()
        sim_index_label = QLabel("Projection Index:")
        self.sim_index_input = QLineEdit()
        self.sim_index_input.setPlaceholderText("e.g., 0")
        self.sim_index_input.returnPressed.connect(self.update_simulation_projection)

        sim_update_button = QPushButton("Show")
        sim_update_button.clicked.connect(self.update_simulation_projection)

        sim_index_layout.addWidget(sim_index_label)
        sim_index_layout.addWidget(self.sim_index_input)
        sim_index_layout.addWidget(sim_update_button)
        left_layout.addLayout(sim_index_layout)

        # Class Averages projection plot
        self.left_plot = PlotCanvas(self, width=5, height=5)
        left_layout.addWidget(self.left_plot)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        left_layout.addWidget(self.progress_label)
        left_layout.addWidget(self.progress_bar)

        main_layout.addLayout(left_layout, 1)

        # ==== RIGHT PANEL ====
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # Display number of available cache rotations
        self.rotation_count_label = QLabel("Available cache rotations: Loading...")
        self.rotation_count_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.rotation_count_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.rotation_count_label)

        # Input for cache rotation index
        index_label = QLabel("Enter Cache Rotation Index:")
        index_label.setFont(QFont("Arial", 10))
        self.index_input = QLineEdit()
        self.index_input.setPlaceholderText("e.g., 3")
        self.index_input.returnPressed.connect(self.update_selected_rotation)

        update_button = QPushButton("Show Rotation")
        update_button.clicked.connect(self.update_selected_rotation)

        right_layout.addWidget(index_label)
        right_layout.addWidget(self.index_input)
        right_layout.addWidget(update_button)

        # Checkboxes and flip buttons container
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(5)

        # J value checkbox and flip button
        j_layout = QHBoxLayout()
        self.j_value_checkbox = QCheckBox("J (handedness)")
        self.j_value_checkbox.setEnabled(True)
        self.j_value_checkbox.setFont(QFont("Arial", 10))

        self.flip_j_button = QPushButton("Flip J")
        self.flip_j_button.clicked.connect(self.flip_j_values)
        self.flip_j_button.setMaximumWidth(80)

        j_layout.addWidget(self.j_value_checkbox)
        j_layout.addStretch()  # This pushes the button to the right
        j_layout.addWidget(self.flip_j_button)
        controls_layout.addLayout(j_layout)

        # R value checkbox and flip button (only shown for T or D symmetries)
        r_layout = QHBoxLayout()
        self.r_value_checkbox = QCheckBox("r (normalizer coset)")
        self.r_value_checkbox.setEnabled(True)
        self.r_value_checkbox.setFont(QFont("Arial", 10))

        self.flip_r_button = QPushButton("Flip r")
        self.flip_r_button.clicked.connect(self.flip_r_values)
        self.flip_r_button.setMaximumWidth(80)

        r_layout.addWidget(self.r_value_checkbox)
        r_layout.addStretch()  # This pushes the button to the right
        r_layout.addWidget(self.flip_r_button)
        controls_layout.addLayout(r_layout)

        # Initially hide r controls
        self.r_value_checkbox.setVisible(False)
        self.flip_r_button.setVisible(False)

        right_layout.addLayout(controls_layout)

        # Selected cache plot
        self.right_plot = PlotCanvas(self, width=5, height=5)
        right_layout.addWidget(QLabel("Selected Cache Rotation Projection"))
        right_layout.addWidget(self.right_plot)
        main_layout.addLayout(right_layout, 1)

    def flip_j_values(self):
        """Flip all J values in cache_selected_data by multiplying by -1"""
        if (self.cache_selected_data is not None and
                'cache_J_indices' in self.cache_selected_data):
            self.cache_selected_data['cache_J_indices'] *= -1
            self.progress_label.setText("Flipped all J values")
        else:
            self.progress_label.setText("No J values to flip")

    def flip_r_values(self):
        """Flip all r values in cache_selected_data by multiplying by -1"""
        if (self.cache_selected_data is not None and
                'cache_selected_r_indices' in self.cache_selected_data):
            self.cache_selected_data['cache_selected_r_indices'] *= -1
            self.progress_label.setText("Flipped all r values")
        else:
            self.progress_label.setText("No r values to flip")

    def load_data(self):
        """Load and process data in a separate thread"""
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_label.setText("Loading data...")

        self.data_processor = DataProcessor(
            sym=self.sym,
            cache_file=self.cache_file,
            simulation_projections=self.simulation_projections,
            cache_selected_data=self.cache_selected_data
        )
        self.data_processor.data_ready.connect(self.on_data_ready)
        self.data_processor.progress_update.connect(self.on_progress_update)
        self.data_processor.start()

    def on_progress_update(self, message):
        """Update progress display"""
        self.progress_label.setText(message)

    def on_data_ready(self, volume, cache_rotations, cache_data, sim_projections, cache_selected_data):
        """Handle processed data from the worker thread"""
        self.volume = volume
        self.cache_rotations = cache_rotations
        self.cache_data = cache_data
        self.sim_projections = sim_projections
        self.cache_selected_data = cache_selected_data

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.progress_label.setText("Data loaded successfully!")

        # Update rotation count displays
        if self.cache_rotations is not None:
            n_rotations = len(self.cache_rotations)
            self.rotation_count_label.setText(f"Available cache rotations: {n_rotations}")

        if self.sim_projections is not None:
            n_sim_projections = len(self.sim_projections)
            self.sim_projection_count_label.setText(f"Total simulation projections: {n_sim_projections}")
        else:
            self.sim_projection_count_label.setText("No simulation projections provided")

        # Validate cache_selected_data length
        if (self.cache_selected_data is not None and
                'cache_selected_inds' in self.cache_selected_data and
                self.sim_projections is not None):
            cache_selected_inds = self.cache_selected_data['cache_selected_inds']
            if len(cache_selected_inds) != len(self.sim_projections):
                self.progress_label.setText(
                    f"Warning: cache_selected_inds length ({len(cache_selected_inds)}) doesn't match simulation projections length ({len(self.sim_projections)})")

        # Show/hide r controls based on symmetry
        if self.sym == "T" or (isinstance(self.sym, str) and self.sym.startswith("D")):
            self.r_value_checkbox.setVisible(True)
            self.flip_r_button.setVisible(True)
        else:
            self.r_value_checkbox.setVisible(False)
            self.flip_r_button.setVisible(False)

        # Show empty plots initially
        self.show_empty_plots()

    def show_empty_plots(self):
        """Show empty plots initially"""
        # Left plot - empty initially
        if self.sim_projections is not None:
            self.left_plot.ax.clear()
            self.left_plot.ax.text(0.5, 0.5, 'Select a simulation projection\nindex to view image',
                                   ha='center', va='center', transform=self.left_plot.ax.transAxes)
            self.left_plot.ax.set_title("Class Averages Projection", fontsize=10)
            self.left_plot.draw()
        else:
            self.left_plot.ax.clear()
            self.left_plot.ax.text(0.5, 0.5, 'No simulation projections\nprovided',
                                   ha='center', va='center', transform=self.left_plot.ax.transAxes)
            self.left_plot.ax.set_title("Class Averages Projections", fontsize=10)
            self.left_plot.draw()

        # Right plot - empty initially
        self.right_plot.ax.clear()
        self.right_plot.ax.text(0.5, 0.5, 'Select a cache rotation index\nto view projection',
                                ha='center', va='center', transform=self.right_plot.ax.transAxes)
        self.right_plot.ax.set_title("Selected Cache Rotation Projection", fontsize=10)
        self.right_plot.draw()

    def update_j_value_checkbox(self, simulation_index):
        """Update the J value checkbox based on the current data arrays"""
        if (self.cache_selected_data is not None and
                'cache_J_indices' in self.cache_selected_data and
                'cache_selected_inds' in self.cache_selected_data and
                self.current_sim_projection_selected):

            cache_J_indices = self.cache_selected_data['cache_J_indices']
            j_value = cache_J_indices[simulation_index]
            # Check if -1, uncheck if 1
            self.j_value_checkbox.setChecked(bool(j_value == -1))

    def update_r_value_checkbox(self, simulation_index):
        """Update the R value checkbox based on the current data arrays"""
        if (self.cache_selected_data is not None and
                'cache_selected_r_indices' in self.cache_selected_data and
                'cache_selected_inds' in self.cache_selected_data and
                self.current_sim_projection_selected and
                (self.sym == "T" or (isinstance(self.sym, str) and self.sym.startswith("D")))):

            cache_r_indices = self.cache_selected_data['cache_selected_r_indices']
            r_value = cache_r_indices[simulation_index]
            # Check if -1, uncheck if 1
            self.r_value_checkbox.setChecked(bool(r_value == -1))

    def get_selection_count(self, simulation_index):
        """Get the selection count for a cache index if available"""
        if (self.cache_selected_data is not None and
                'cache_selected_count' in self.cache_selected_data and
                'cache_selected_inds' in self.cache_selected_data):

            cache_selected_count = self.cache_selected_data['cache_selected_count']
            return cache_selected_count[simulation_index]
        return None

    def update_simulation_projection(self):
        """Update the simulation projection display"""
        try:
            if self.sim_projections is None:
                self.progress_label.setText("No simulation projections available")
                return

            index = int(self.sim_index_input.text())
            n_sim_projections = len(self.sim_projections)

            if 0 <= index < n_sim_projections:
                self.current_sim_projection_index = index
                self.current_sim_projection_selected = True

                # Update checkboxes to reflect current data array values
                self.update_j_value_checkbox(index)
                self.update_r_value_checkbox(index)

                selection_count = self.get_selection_count(index)

                # Get the specific simulation projection
                projection_image = self.sim_projections[index]

                # Display the projection
                self.left_plot.plot_image(
                    projection_image,
                    f"Class Average #{index} Projection"
                )

                # Auto-update the corresponding cache rotation if cache_selected_data is available
                if (self.cache_selected_data is not None and
                        'cache_selected_inds' in self.cache_selected_data and
                        index < len(self.cache_selected_data['cache_selected_inds'])):

                    corresponding_cache_index = self.cache_selected_data['cache_selected_inds'][index]
                    # Update the right plot input field and display
                    self.index_input.setText(str(corresponding_cache_index))
                    # Set simulation projection as selected when auto-updating
                    self.current_sim_projection_selected = True
                    self.update_selected_rotation_by_index(corresponding_cache_index, selection_count)
                    self.progress_label.setText(
                        f"Showing simulation projection {index} and corresponding cache rotation {corresponding_cache_index}")
                else:
                    self.progress_label.setText(f"Showing simulation projection {index}")
                    # Set simulation projection as selected even if no corresponding cache
                    self.current_sim_projection_selected = True

            else:
                self.progress_label.setText(f"Index {index} out of range (0-{n_sim_projections - 1})")

        except ValueError:
            self.progress_label.setText("Please enter a valid integer for simulation projection")

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

                # Get the specific rotation matrix
                rotation_matrix = self.cache_rotations[index]

                # Project the volume using this rotation
                self.progress_label.setText(f"Computing projection for cache rotation {index}...")

                try:
                    # Convert to proper format for ASPIRE if needed
                    if hasattr(rotation_matrix, 'shape') and rotation_matrix.shape == (3, 3):
                        # Apply transformations based on checkbox states
                        display_matrix = rotation_matrix.copy()

                        # Apply r transformation if checkbox is checked and visible
                        if self.r_value_checkbox.isVisible() and self.r_value_checkbox.isChecked():
                            normalizer_group = group_elements(normalizer(self.sym))
                            r = normalizer_group[-1]
                            display_matrix = r @ display_matrix

                        # Apply J transformation if checkbox is checked
                        if self.j_value_checkbox.isChecked():
                            display_matrix = multi_Jify(display_matrix)

                        # Reshape for ASPIRE (expects shape [1, 3, 3])
                        rot_for_projection = display_matrix.reshape(1, 3, 3).astype(np.float32)

                        projection = self.volume.project(rot_for_projection)

                        # Handle ASPIRE Image object
                        if hasattr(projection, 'asnumpy'):
                            proj_array = projection.asnumpy()
                            if len(proj_array.shape) >= 3:
                                proj_image = proj_array[0]  # First (and only) projection
                            else:
                                proj_image = proj_array
                        else:
                            proj_image = projection

                        # Add selection count text if available and simulation projection was chosen
                        count_text = ""
                        if selection_count is not None and self.current_sim_projection_selected:
                            count_text = f"Selection count: {selection_count} (of {len(self.sim_projections) - 1})"

                        # Display the projection
                        title = f"Cache Rotation #{index} Projection"
                        self.right_plot.plot_image(proj_image, title, "", count_text)

                    else:
                        self.progress_label.setText(f"Invalid rotation matrix shape: {rotation_matrix.shape}")

                except Exception as e:
                    self.progress_label.setText(f"Error computing projection: {str(e)}")
                    # Show a dummy projection for testing
                    dummy_proj = np.random.rand(64, 64)
                    self.right_plot.plot_image(dummy_proj, f"Dummy Cache Projection {index}")

            else:
                self.progress_label.setText(f"Index {index} out of range (0-{n_rotations - 1})")
        else:
            self.progress_label.setText("Data not loaded yet")


def create_projection_viewer_class_avgs(sym='T', cache_file=None, simulation_projections=None, cache_selected_data=None):
    """
    Factory function to create and show the rotation viewer.

    Args:
        sym (str): Symmetry type ('T', 'O', 'I', etc.)
        cache_file (str): Path to the cache file containing rotation matrices
        simulation_projections (np.ndarray): Array of simulation projection images with shape (n, size, size)
        cache_selected_data (dict): Dictionary containing:
            - 'cache_selected_inds': Array of cache indices corresponding to each simulation projection
            - 'cache_J_indices': Array of J values (-1 or 1) for each simulation projection
            - 'cache_selected_count': Array of selection counts for each projection
            - 'cache_selected_r_indices': Array of r values (-1 or 1) for normalizer coset (T/D symmetries only)

    Returns:
        ProjectionViewer: The created viewer widget
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    viewer = ProjectionViewer(sym=sym, cache_file=cache_file, simulation_projections=simulation_projections,
                            cache_selected_data=cache_selected_data)
    viewer.show()
    app.exec_()
    return viewer


# Example usage and main execution
if __name__ == "__main__":
    # Example parameters
    sym = 'T'
    cache_file = ROTATIONS_CACHE_DIR / "cache_T_symmetry_resolution_150_ntheta_360_view_direc_0.996_in_plane_5.pkl"

    # Example simulation projections (dummy data for testing - replace with actual projection images)
    # Shape should be (n_projections, image_size, image_size)
    simulation_projections = np.random.rand(25, 70, 70)

    # Example cache_selected_data with the new dictionary format
    cache_selected_data = {
        "cache_selected_inds": np.random.randint(0, 100, 25),  # Random indices between 0-99
        "cache_J_indices": np.random.choice([-1, 1], 25),  # Random J values (-1 or 1)
        "cache_selected_count": np.random.randint(1, 10, 25),  # Random selection counts
        "cache_selected_r_indices": np.random.choice([-1, 1], 25),  # Random r values (-1 or 1)
    }

    create_projection_viewer_class_avgs(sym=sym, cache_file=cache_file, simulation_projections=simulation_projections,
                           cache_selected_data=cache_selected_data)