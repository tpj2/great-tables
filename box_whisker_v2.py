import pandas as pd
import numpy as np

# Import necessary components from great_tables
from great_tables import GT, html, md

# --- Step 1: Create Sample Data ---
# Generate a date range
dates = pd.date_range(start='2024-01-01', end='2025-03-25', freq='B')  # Business days

# Define tenors
tenors = ['1Y', '2Y', '3Y', '5Y', '10Y', '20Y', '30Y']

# Generate semi-realistic yield curve data (adding some noise and trend)
n_dates = len(dates)
n_tenors = len(tenors)

# Base yields (can drift over time)
base_yields = np.linspace(1.5, 4.0, n_tenors)  # Start with an upward sloping curve
data = np.zeros((n_dates, n_tenors))

for i in range(n_dates):
    noise = np.random.normal(0, 0.05, n_tenors)
    drift = i * 0.0001  # Slight upward trend over the period
    current_base = base_yields + drift
    data[i, :] = current_base + noise

# Ensure yields don't go below a certain floor (e.g., 0)
data = np.maximum(data, 0.1)

# Create DataFrame
df_yields = pd.DataFrame(data, index=dates, columns=tenors)
df_yields.index.name = 'Date'

print("--- Sample Yield Data (First 5 rows) ---")
print(df_yields.head())
print("\n")


# --- Step 2: Define Enhanced SVG Boxplot Function ---
def create_sideways_boxplot_svg(
    data_vector,
    current_value,
    width=200,        # Increased width to accommodate min/max labels
    height=30,        # Increased height for better visibility
    box_height=16,     # Height of the box itself
    line_color="#666666", # Default line color (grey)
    box_color="#AAAAFF",  # Default box color (light blue/purple)
    median_color="#000000", # Default median line color (black)
    dot_color="#FF0000",   # Default dot color (red)
    dot_size=6,        # Radius of the dot
    triangle_color="#FFD700", # Yellow color for average triangle
    triangle_size=8,   # Size of the triangle
    show_minmax_labels=True, # Whether to show min/max labels
    label_font_size=8  # Font size for min/max labels
):
    """
    Generates an inline SVG string representing a sideways box-and-whisker plot.
    Now includes:
    - Yellow triangle for the average
    - Min/Max value labels at the ends of the whiskers
    """
    try:
        # 1. Calculate necessary statistics (handle potential NaNs)
        min_val = np.nanmin(data_vector)
        max_val = np.nanmax(data_vector)
        q1 = np.nanpercentile(data_vector, 25)
        median = np.nanpercentile(data_vector, 50)
        q3 = np.nanpercentile(data_vector, 75)
        mean_val = np.nanmean(data_vector)  # Calculate average
        current_val_clean = float(current_value) # Ensure it's a float

        # Handle edge case where all data is the same
        if min_val == max_val:
            x_min = x_q1 = x_median = x_q3 = x_max = x_current = x_mean = width / 2
        else:
            # 2. Scale values to SVG coordinates (add padding for labels)
            padding = 25 if show_minmax_labels else 5  # More padding for labels
            scale_width = width - 2 * padding
            data_range = max_val - min_val

            def scale_value(val):
                clamped_val = max(min_val, min(val, max_val))
                return padding + ((clamped_val - min_val) / data_range) * scale_width

            x_min = scale_value(min_val)
            x_max = scale_value(max_val)
            x_q1 = scale_value(q1)
            x_median = scale_value(median)
            x_q3 = scale_value(q3)
            x_current = scale_value(current_val_clean)
            x_mean = scale_value(mean_val)  # Scale the average position

        # 3. Define SVG elements
        y_center = height / 2
        box_top = y_center - (box_height / 2)

        # Start building SVG
        svg_elements = []

        # Whisker line
        whisker_line = f'<line x1="{x_min}" y1="{y_center}" x2="{x_max}" y2="{y_center}" stroke="{line_color}" stroke-width="1"/>'
        svg_elements.append(whisker_line)

        # Box (prevent zero-width box if q1==q3)
        box_width = max(1, x_q3 - x_q1)
        box_rect = f'<rect x="{x_q1}" y="{box_top}" width="{box_width}" height="{box_height}" fill="{box_color}" stroke="{line_color}" stroke-width="0.5"/>'
        svg_elements.append(box_rect)

        # Median line
        median_line = f'<line x1="{x_median}" y1="{box_top}" x2="{x_median}" y2="{box_top + box_height}" stroke="{median_color}" stroke-width="1.5"/>'
        svg_elements.append(median_line)

        # Current value dot
        current_dot = f'<circle cx="{x_current}" cy="{y_center}" r="{dot_size}" fill="{dot_color}"/>'
        svg_elements.append(current_dot)

        # NEW: Yellow triangle for average
        triangle_points = f"{x_mean},{y_center-triangle_size} {x_mean-triangle_size},{y_center+triangle_size} {x_mean+triangle_size},{y_center+triangle_size}"
        average_triangle = f'<polygon points="{triangle_points}" fill="{triangle_color}" stroke="#B8860B" stroke-width="0.5"/>'
        svg_elements.append(average_triangle)

        # NEW: Min/Max labels
        if show_minmax_labels:
            min_label = f'<text x="{x_min}" y="{height-2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{label_font_size}" fill="{line_color}">{min_val:.2f}</text>'
            max_label = f'<text x="{x_max}" y="{height-2}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{label_font_size}" fill="{line_color}">{max_val:.2f}</text>'
            svg_elements.append(min_label)
            svg_elements.append(max_label)

        # 4. Assemble SVG string with accessibility
        svg_title = f"Min: {min_val:.2f}, Q1: {q1:.2f}, Median: {median:.2f}, Mean: {mean_val:.2f}, Q3: {q3:.2f}, Max: {max_val:.2f}, Current: {current_val_clean:.2f}"
        svg = f'<svg width="{width}" height="{height}" style="vertical-align: middle;" role="img" aria-label="Sideways box plot with average triangle and min/max labels">'
        svg += f'<title>{svg_title}</title>' # Tooltip / screen reader info
        svg += ''.join(svg_elements)
        svg += '</svg>'

        return svg

    except Exception as e:
        print(f"Error generating SVG for value {current_value}: {e}")
        return "Error creating plot"


# --- Step 3: Create Summary Table using great-tables ---

# Get the latest yields ("Today's" value)
latest_yields = df_yields.iloc[-1]
latest_date_str = latest_yields.name.strftime('%Y-%m-%d')
historical_start_date_str = df_yields.index.min().strftime('%Y-%m-%d')

# Prepare data for the summary table
summary_data_for_gt = []
for tenor in tenors:
    historical_data = df_yields[tenor]
    current_yield = latest_yields[tenor]

    # Generate the enhanced SVG for this tenor's distribution
    svg_plot_str = create_sideways_boxplot_svg(
        historical_data,
        current_yield
    )

    summary_data_for_gt.append({
        "Tenor": tenor,
        "Yield": current_yield,
        "Distribution": svg_plot_str
    })

# Create the summary DataFrame
summary_df_gt = pd.DataFrame(summary_data_for_gt)

# --- Create and format the great_tables object ---
print("--- Creating great_tables object ---")
gt = GT(data=summary_df_gt)

# Add Title and Subtitle
gt = gt.tab_header(
    title=md("**US Treasury Yield Curve Summary**"), 
    subtitle=md(f"_Yields as of {latest_date_str} compared to historical distribution ({historical_start_date_str} to {latest_date_str})_")
)

# Format the Yield column as percentage
gt = gt.fmt_number(
    columns="Yield",
    decimals=2,
    pattern="{x}%" 
)

# Relabel columns for better presentation
gt = gt.cols_label(
    Tenor="Tenor",
    Yield="Current Yield",
    Distribution="Distribution"
)

# Align columns
gt = gt.cols_align(
    align="center",
    columns=["Tenor", "Yield", "Distribution"]
)

# Display the table
print(gt)