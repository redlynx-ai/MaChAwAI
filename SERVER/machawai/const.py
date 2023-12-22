"""
Global constants.
"""

# ----------------------------
# --- TEST SETUP CONSTANTS ---
# ----------------------------

MOTOR = "MOTOR"
ENCODER = "ENCO"
EXTENSO = "EXTENSO"

ACCEPTED_REF_DISPLACEMENT_LOAD = [MOTOR, ENCODER, EXTENSO]
ACCEPTED_REF_PARAM_STRAIN = [MOTOR, ENCODER, EXTENSO]

RP02 = "Rp02"
RP05 = "Rp05"
RP1 = "Rp1"
RP2 = "Rp2"

LINEARITY_DEV_METHODS = [RP02, RP05, RP1, RP2]

# ---------------------------
# --- WORKBOOKS CONSTANTS ---
# ---------------------------

# >>> Sheets name
#
RAW = "(1) RAW"
ELAB = "(2) ELAB"

# >>> Data col. name
#
TIME_COL = "TIME"
DISP_COL = "DISP"
LOAD_COL = "LOAD"
EXTS_COL = "EXTS"

# >>> Specimen properties table
#
SPROP_START = "G13"
SPROP_END = "I17"

# >>> Setup table
#
SETUP_START = "I24"
SETUP_END = "I29"

# >>> Cut and Offset
TAIL_P_CELL = "I38"
FOOT_OFFSET_CELL = "I43"

# >>> Linear Section
BOTTOM_CUTOUT_CELL = "G57"
UPPER_CUTOUT_CELL = "H57"