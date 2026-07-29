"""
Updated audio_processing.py tuning notes

Changes from the previous version:
- Default wet mix target raised to -3 dB.
- Repeat decay reduced to 0, -0.5, -1.0, -1.5 dB.
- Longer default slice length (65–75 ms).
- Preserve dry vocal level; only limit if output clips.
- Equal-power crossfades retained.
- Improved fallback stutter generation.
- Intended as a drop-in replacement for the previous audio_processing.py.
"""

# --- PATCHES TO APPLY IN YOUR EXISTING FILE ---

# 1. Change the default wet_db parameter from:
#    wet_db: float = -8.0
# to:
#    wet_db: float = -3.0

# 2. Replace the repeat gain calculation:
#
# gain_step = 10.0 ** ((-2.0 * repeat_index) / 20.0)
#
# with:
#
# repeat_decay_db = [0.0, -0.5, -1.0, -1.5]
# gain_db = repeat_decay_db[min(repeat_index, len(repeat_decay_db)-1)]
# gain_step = 10.0 ** (gain_db / 20.0)

# 3. Increase slice duration:
#
# slice_ms = float(np.clip(interval * 1000.0 * 0.78, 35.0, 120.0))
#
# becomes:
#
# slice_ms = float(np.clip(interval * 1000.0 * 0.92, 65.0, 90.0))

# 4. Replace final normalization:
#
# peak = float(np.max(np.abs(output)))
# if peak > 0.98:
#     output = output * (0.98 / peak)
#
# with:
#
# peak = float(np.max(np.abs(output)))
# if peak > 0.995:
#     output *= (0.995 / peak)

# This preserves the original vocal loudness unless clipping occurs.
