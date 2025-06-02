# -*- coding: utf-8 -*-
"""
Created on Fri May  9 16:38:27 2025

@author: andre redondo
"""

import streamlit as st
import pandas as pd
import random
from typing import List, Tuple
import numpy as np

# Define constants
ROWS = list("ABCDEFGH")
COLUMNS = list(range(1, 13))
POS_CONTROLS = ["A1", "A2", "A3", "A10", "A11", "A12", "D4", "D5", "D6", "H1", "H2", "H3"]
NEG_CONTROLS = ["H10", "H11", "H12"]
FIXED_WELLS = set(POS_CONTROLS + NEG_CONTROLS)
COLOR_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
    "#CC79A7", "#999999", "#DDAA33", "#004488", "#BB5566", "#AAAA00"
]

SEED = 42  # or use any number / st.sidebar input
random.seed(SEED)
np.random.seed(SEED)

def generate_plate_wells():
    return [f"{r}{c}" for r in ROWS for c in COLUMNS]

def column_triplets():
    return [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

def add_controls_to_plate(df_plate, plate_id):
    for well in POS_CONTROLS:
        df_plate = pd.concat([df_plate, pd.DataFrame([{
            "Plate": f"Plate {plate_id}",
            "Well": well,
            "Vector": "POS_CTRL",
            "Condition": "POS_CTRL",
            "Timepoint": "",
            "Replicate": ""
        }])], ignore_index=True)
    for well in NEG_CONTROLS:
        df_plate = pd.concat([df_plate, pd.DataFrame([{
            "Plate": f"Plate {plate_id}",
            "Well": well,
            "Vector": "NEG_CTRL",
            "Condition": "NEG_CTRL",
            "Timepoint": "",
            "Replicate": ""
        }])], ignore_index=True)
    return df_plate

def randomize_across_plates(df: pd.DataFrame, name_map: pd.DataFrame) -> pd.DataFrame:
    mode = st.session_state.get("timepoint_assignment_mode", "Sequential")

    # Step 1: Create grouped triplicates based on (Vector, Condition)
    triplicates = []
    grouped = df.groupby(["Vector", "Condition"])

    for (vec, cond), group in grouped:
        timepoints = list(group["Timepoint"])

        if mode == "Randomized":
            random.shuffle(timepoints)  # Shuffle timepoints only within this group

        for tp in timepoints:
            triplicates.append((vec, cond, tp))

    # Optional: Shuffle full triplicate order across plate (plate-level randomization)
    random.shuffle(triplicates)  # You can skip this if you want full sequential layout by group

    # Step 2: Assign triplicates one by one to sequential plates
    final_layout = pd.DataFrame()
    plate_id = 1
    plate_capacity = len(ROWS) * len(COLUMNS)
    used_wells = set(FIXED_WELLS)

    layout = {}

    for trip in triplicates:
        placed = False
        attempt_rows = ROWS.copy()
        random.shuffle(attempt_rows)

        while not placed:
            if len(used_wells) >= plate_capacity:
                df_plate = pd.DataFrame(layout.values())
                df_plate = add_controls_to_plate(df_plate, plate_id)
                final_layout = pd.concat([final_layout, df_plate], ignore_index=True)

                # Start a new plate
                plate_id += 1
                used_wells = set(FIXED_WELLS)
                layout = {}

            triplet_groups = column_triplets()
            random.shuffle(triplet_groups)

            for row in attempt_rows:
                for cols in triplet_groups:
                    wells = [f"{row}{c}" for c in cols]
                    if all(w not in used_wells for w in wells):
                        for well, rep in zip(wells, ["rep1", "rep2", "rep3"]):
                            layout[well] = {
                                "Plate": f"Plate {plate_id}",
                                "Well": well,
                                "Vector": trip[0],
                                "Condition": trip[1],
                                "Timepoint": trip[2],
                                "Replicate": rep
                            }
                            used_wells.add(well)
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                raise RuntimeError("Failed to place triplicate on current plate or start new one.")

    # Save the final plate
    df_plate = pd.DataFrame(layout.values())
    df_plate = add_controls_to_plate(df_plate, plate_id)
    final_layout = pd.concat([final_layout, df_plate], ignore_index=True)

    return final_layout




    # Step 4: Map to real names
    name_lookup = name_map.set_index("Local Name")["Real Name"].to_dict()
    final_layout["Vector Name"] = final_layout["Vector"].map(name_lookup).fillna("")
    final_layout["Condition Name"] = final_layout["Condition"].map(name_lookup).fillna("")
    return final_layout

def get_user_input_table() -> pd.DataFrame:
    print("Select input mode:")
    print("1. Generate factorial experimental design")
    print("2. Load prepared table from Excel")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        return interactive_design()  # We'll define this below
    elif mode == "2":
        file_path = input("Enter the path to the Excel file (e.g., input.xlsx): ").strip()
        try:
            df = pd.read_excel(file_path)
            required_columns = {"Vector", "Condition", "Timepoint"}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"The Excel file must contain the following columns: {required_columns}")
            return df
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            exit(1)
    else:
        print("Invalid choice. Please enter 1 or 2.")
        exit(1)

# --- Initialize session state variables ---
if "pending_step3" not in st.session_state:
    st.session_state.pending_step3 = False

if "timepoint_assignment_mode" not in st.session_state:
    st.session_state.timepoint_assignment_mode = "Sequential"

st.set_page_config(page_title="PCP Plate Randomizer", layout="wide")
st.title("🌱 PCP Experimental Design Tool")

    
# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "df" not in st.session_state:
    st.session_state.df = None
if "name_map" not in st.session_state:
    st.session_state.name_map = {}

# Step 1: Define experiment
if st.session_state.step == 1:
    st.header("Step 1: Choose Input Mode")

    input_mode = st.radio("How would you like to start?", ["Generate factorial design", "Paste prepared table"])

    if input_mode == "Generate factorial design":
        with st.form("experiment_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                num_vectors = st.number_input("Number of vectors", min_value=1, step=1, value=2)
            with col2:
                num_conditions = st.number_input("Number of conditions", min_value=1, step=1, value=2)
            with col3:
                num_timepoints = st.number_input("Number of timepoints", min_value=1, step=1, value=2)

            submitted = st.form_submit_button("Generate Combinations")

        if submitted:
            vectors = [f"V{i+1}" for i in range(num_vectors)]
            conditions = [f"C{j+1}" for j in range(num_conditions)]
            timepoints = [f"T{k+1}" for k in range(num_timepoints)]

            data = []
            for v in vectors:
                for c in conditions:
                    for t in timepoints:
                        data.append({"Vector": v, "Condition": c, "Timepoint": t})

            st.session_state.df = pd.DataFrame(data)

            # Initialize name mapping table
            mapping_data = (
                [{"Type": "Vector", "Local Name": v, "Real Name": ""} for v in vectors] +
                [{"Type": "Condition", "Local Name": c, "Real Name": ""} for c in conditions]
            )
            st.session_state.name_map = pd.DataFrame(mapping_data)

            st.session_state.step = 2
            st.rerun()

    else:  # Paste prepared table
        st.markdown("### Paste your data into this table")
        st.markdown("Use **Ctrl+V** to paste from Excel. The table must contain exactly these columns: `Vector`, `Condition`, `Timepoint`.")

        expected_columns = ["Vector", "Condition", "Timepoint"]
        empty_df = pd.DataFrame(columns=expected_columns)
        edited_df = st.data_editor(empty_df, num_rows="dynamic", key="paste_table_editor")

        # Create a transition flag if it doesn't exist
        if "pending_step2_transition" not in st.session_state:
            st.session_state.pending_step2_transition = False

        # Handle button click
        confirm_clicked = st.button("✅ Confirm Table")

        if confirm_clicked:
            if edited_df.isnull().any().any():
                st.warning("⚠️ Please fill in all cells before continuing.")
            elif set(edited_df.columns) != set(expected_columns):
                st.error("❌ Table must have exactly these columns: Vector, Condition, Timepoint.")
            else:
                st.session_state.df = edited_df.dropna()

                vectors = sorted(st.session_state.df["Vector"].unique())
                conditions = sorted(st.session_state.df["Condition"].unique())

                mapping_data = (
                    [{"Type": "Vector", "Local Name": v, "Real Name": ""} for v in vectors] +
                    [{"Type": "Condition", "Local Name": c, "Real Name": ""} for c in conditions]
                )
                st.session_state.name_map = pd.DataFrame(mapping_data)

                # Set the transition flag and rerun
                st.session_state.pending_step2_transition = True
                st.rerun()

        # Handle the actual transition in a separate run
        if st.session_state.pending_step2_transition:
            st.session_state.step = 2
            st.session_state.pending_step2_transition = False
            st.rerun()




# Step 2: Edit combinations and names
if st.session_state.step == 2:
    st.header("Step 2: Customize Names and Edit Experimental Conditions")
    timepoint_assignment_mode = st.selectbox(
        "How should timepoints be assigned to triplicates?",
        ["Sequential", "Randomized"],
        key="timepoint_assignment_mode"
    )
    
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ✏️ Name Mapping Table")
    
        edited_name_map = st.data_editor(
            st.session_state.name_map,
            num_rows="dynamic",
            use_container_width=True,
            key="step2_name_editor"
        )
    
    with col2:
        st.markdown("### 🧪 Experimental Design Table")
    
        edited_df = st.data_editor(
            st.session_state.df,
            num_rows="dynamic",
            use_container_width=True,
            key="step2_design_editor"
        )


    st.divider()
    col3, col4 = st.columns([1, 3])
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.step = 1
            st.rerun()
    if "pending_step3" not in st.session_state:
        st.session_state.pending_step3 = False
    
    with col4:
        if st.button("✅ Approve and Continue to Plate Layout"):
            # Save name map
            st.session_state.name_map = edited_name_map
    
            # Apply renaming to experimental design table
            old_names = st.session_state.name_map["Local Name"].values
            new_names = edited_name_map["Local Name"].values
            df_updated = edited_df.copy()
            for old, new in zip(old_names, new_names):
                if old != new:
                    df_updated = df_updated.replace(old, new)
    
            st.session_state.df = df_updated
            st.session_state.pending_step3 = True
            st.rerun()

if st.session_state.pending_step3:
    st.session_state.step = 3
    st.session_state.pending_step3 = False
    st.rerun()

def render_plate_grid_colored(df, plate_name, use_real_names=False):
    st.markdown(f"### 🧫 {plate_name}")
    grid = []
    color_map = {}
    row_labels = list("ABCDEFGH")
    col_labels = [str(i) for i in range(1, 13)]

    # Replace local names with real names if selected
    if use_real_names:
        name_dict = dict(zip(st.session_state.name_map["Local Name"], st.session_state.name_map["Real Name"]))
        df = df.copy()
        df["Vector"] = df["Vector"].replace(name_dict)
        df["Condition"] = df["Condition"].replace(name_dict)

    # Assign color to each unique (Vector, Condition) pair
    combos = df[["Vector", "Condition"]].drop_duplicates()
    for i, (_, row) in enumerate(combos.iterrows()):
        key = (row["Vector"], row["Condition"])
        color_map[key] = COLOR_PALETTE[i % len(COLOR_PALETTE)]

    # Create 8x12 grid content
    for r in row_labels:
        row = []
        for c in col_labels:
            well = f"{r}{c}"
            well_data = df[df["Well"] == well]
            if not well_data.empty:
                row_data = well_data.iloc[0]
                if row_data["Vector"] == "POS_CTRL":
                    bg_color = "#FF69B4"  # Hot pink
                    label = "<b>POS CTRL</b>"
                elif row_data["Vector"] == "NEG_CTRL":
                    bg_color = "#9370DB"  # Purple
                    label = "<b>NEG CTRL</b>"
                else:
                    label = f'{row_data["Vector"]}<br>{row_data["Condition"]}<br>{row_data["Timepoint"]}'
                    key = (row_data["Vector"], row_data["Condition"])
                    bg_color = color_map.get(key, "#FFFFFF")
                row.append(f'<div style="background-color:{bg_color}; padding:4px; font-size:10px; text-align:center;">{label}</div>')
            else:
                row.append("")
        grid.append(row)

    styled_table = pd.DataFrame(grid, index=[f"{i+1}" for i in range(8)], columns=col_labels)

    st.markdown(styled_table.to_html(escape=False), unsafe_allow_html=True)

    # Show legend
    st.markdown("### 🧷 Color Legend")
    for key, color in color_map.items():
        v, c = key
        st.markdown(
            f'<div style="display:inline-block; background-color:{color}; width:15px; height:15px; margin-right:5px;"></div> {v} | {c}',
            unsafe_allow_html=True
        )


# Step 3: Plate Layout and Randomization
if st.button("⬅️ Go Back to Edit Design"):
    st.session_state.step = 2
    st.rerun()
if st.session_state.step == 3:
    st.header("Step 3: Plate Layout and Randomization")
    st.subheader("Final Experimental Design")
    st.dataframe(st.session_state.df, use_container_width=True)

    st.subheader("Label Map")
    st.dataframe(st.session_state.name_map, use_container_width=True)

    st.divider()
    st.subheader("🔀 Generating Plate Layout...")

    try:
        # Call the randomization function
        final_layout = randomize_across_plates(st.session_state.df, st.session_state.name_map)

        num_plates = final_layout["Plate"].nunique()
        st.success(f"✅ Layout successfully generated across {num_plates} plate(s)!")
        st.session_state.final_layout = final_layout

        def render_plate_grid(df, plate_name):
            st.markdown(f"### 🧫 {plate_name}")
            grid = np.full((8, 12), "", dtype=object)
            row_idx = {r: i for i, r in enumerate("ABCDEFGH")}
            col_idx = {str(c): i for i, c in enumerate(range(1, 13))}
        
            for _, row in df.iterrows():
                well = row["Well"]
                r, c = well[0], well[1:]
                i, j = row_idx[r], col_idx[c]
                if row["Vector"] == "POS_CTRL":
                    label = "POS_CTRL"
                elif row["Vector"] == "NEG_CTRL":
                    label = "NEG_CTRL"
                else:
                    label = f'{row["Vector"]}\n{row["Condition"]}\n{row["Timepoint"]}'
                grid[i, j] = label
        
            plate_df = pd.DataFrame(grid, index=list("ABCDEFGH"), columns=[str(i) for i in range(1, 13)])
            st.dataframe(plate_df, height=300)
        
        # Group by plate and display grids
        st.markdown("## 🧪 Plate Layout Visualization")

        use_real_names = st.checkbox("🔄 Use real names instead of local names", value=False)
        
        for plate in final_layout["Plate"].unique():
            plate_data = final_layout[final_layout["Plate"] == plate]
            render_plate_grid_colored(plate_data, plate, use_real_names=use_real_names)



        st.success("✅ Layout successfully generated!")
        st.dataframe(final_layout, use_container_width=True)

        # Add download button
        csv = final_layout.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Plate Layout as CSV", data=csv, file_name="plate_layout.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error during plate layout generation: {e}")


    if st.button("Go to Step 4: View Plate Layout by Timepoint"):
        st.session_state.step = 4
        st.rerun()

# Step 4: View Plate Layout by Timepoint
if st.session_state.step == 4:
    st.header("Step 4: Filter by Timepoint")

    timepoints = sorted(st.session_state.final_layout["Timepoint"].unique())
    selected_timepoint = st.selectbox("🕒 Select a Timepoint to View", timepoints)

    filtered = st.session_state.final_layout[
        st.session_state.final_layout["Timepoint"] == selected_timepoint
    ]

    st.markdown(f"## 🧪 Plate Layouts for Timepoint: {selected_timepoint}")
    use_real_names = st.checkbox("🔄 Use real names instead of local names", value=False, key="real_names_tp4")

    for plate in filtered["Plate"].unique():
        plate_data = filtered[filtered["Plate"] == plate]
        render_plate_grid_colored(plate_data, plate, use_real_names=use_real_names)

    st.markdown("## 📋 Data Table for This Timepoint")
    st.dataframe(filtered, use_container_width=True)

    # Download filtered layout
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download This Timepoint's Layout as CSV",
        data=csv,
        file_name=f"plate_layout_{selected_timepoint}.csv",
        mime="text/csv"
    )

    st.divider()
    if st.button("⬅️ Back to Full Layout (Step 3)"):
        st.session_state.step = 3
        st.rerun()


#seed as an input
#different color settings
#randomize triplicates
