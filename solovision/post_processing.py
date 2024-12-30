import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics.utils import  MACOS, WINDOWS
import matplotlib.ticker as mticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def tracking_plot(
    result=None, x_data=None, y_data=None, output_path=None, fps=30, line_width=2, max_points=45, 
    video_writer=None, init_only=False, frame=0
):
    """
    Optimized version of tracking plot function.
    Handles graph generation and optional video saving.

    Args:
        result: Current frame result.
        x_data (list): List of x-axis (frame numbers).
        y_data (list): List of y-axis (track counts).
        output_path (str): Directory path to save the video.
        fps (int): Video frames per second.
        max_points (int): Maximum points on the graph.
        line_width (int): Line width of the graph.
        video_writer: Existing cv2.VideoWriter object.
        init_only (bool): Initializes video writer if True.
        frame (int): Current frame number.

    Returns:
        video_writer (cv2.VideoWriter): Initialized writer if init_only is True.
        graph_image (np.ndarray): Generated graph image.
    """
    # Initialize video writer only
    if init_only:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        clip_path = f"{output_path}/tracking_graph"
        suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
        video_writer = cv2.VideoWriter(
            filename=str(Path(clip_path).with_suffix(suffix)),
            fourcc=cv2.VideoWriter_fourcc(*fourcc),
            fps=fps,
            frameSize=(1920, 1080)
        )
        return video_writer

    # Update data
    frame_count = frame
    num_objects = len(result.boxes) if hasattr(result, 'boxes') else len(result.track_ids)
    x_data.append(frame_count)
    y_data.append(num_objects)
    x_data, y_data = x_data[-max_points:], y_data[-max_points:]

    # Determine y-axis limits
    max_y_value = max(y_data) if y_data else 1
    min_y_value = min(y_data) if y_data else 0
    y_upper_limit = max_y_value + max(1, int(max_y_value * 0.2))
    y_lower_limit = max(0, min_y_value - max(1, int(max_y_value * 0.3)))

    # Set Matplotlib style
    plt.style.use("ggplot")  # Use a modern style like 'ggplot'

    # Create a styled plot
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax.clear()
    ax.set_facecolor("#FDFDFD")  # Light background color
    ax.grid(visible=True, color="gray", linestyle="--", alpha=0.5)  # Add subtle gridlines

    # Plot line with markers
    ax.plot(
        x_data, y_data, color="#FF5733", linestyle="-", linewidth=line_width,
        marker="o", markersize=line_width * 5, markerfacecolor="#C70039", markeredgecolor="black"
    )

    # Highlight the last point with annotation
    ax.annotate(
        f"{y_data[-1]}", xy=(x_data[-1], y_data[-1]), xytext=(x_data[-1], y_data[-1] + 0.5),
        fontsize=18, color="black", arrowprops=dict(edgecolor="black", arrowstyle="->")
    )

    # Set titles and labels with custom font sizes and colors
    ax.set_title("Solovision Analytics - Object Tracking", fontsize=28, color="#333333", fontweight="bold")
    ax.set_xlabel("Frame#", fontsize=20, color="#555555")
    ax.set_ylabel("Track ID Count", fontsize=20, color="#555555")

    # Customize ticks
    ax.tick_params(axis="x", labelsize=15, colors="black")
    ax.tick_params(axis="y", labelsize=15, colors="black")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Set dynamic y-limits
    ax.set_ylim(y_lower_limit, y_upper_limit)

    # Add a dynamic legend
    ax.legend(["Number of Tracks"], loc="upper left", fontsize=15, frameon=True, edgecolor="gray")

    # Convert to OpenCV image
    canvas = FigureCanvas(fig)
    canvas.draw()
    graph_image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(fig.canvas.get_width_height()[::-1] + (3,))
    graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
    plt.close(fig)

    # Write graph image to video
    if video_writer:
        video_writer.write(graph_image)

    return graph_image, x_data, y_data
