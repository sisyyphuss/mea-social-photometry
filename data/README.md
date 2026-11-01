# Data Directory Guide

This pipeline expects dual-channel fiber photometry data and 3-chamber behavioral tracking coordinates.

## Expected Formats

1. **Photometry Raw Signals (`Data_2.zip`):**
   - File path inside archive: `Data_2/Social/{young|old}/Fluorescence_{Mouse_ID}.csv`
   - Columns: `TimeStamp`, `CH1-470`, `CH1-410`, `CH2-470`, `CH2-410`, `Events`
   - `CH1`: Left Medial Amygdala (470 nm activity, 410 nm isosbestic reference)
   - `CH2`: Right Medial Amygdala (470 nm activity, 410 nm isosbestic reference)
   - `Events`: Behavioral markers (`Social_Start`, `Social_End`, `Non-Social_Start`, `Non-Social_End`)

2. **Behavioral Tracking Data:**
   - File path: `results_social/individual/Social/{Young|Old}/{Mouse_ID}/synchronized_tracking_photometry.csv`
   - Columns: `x_px`, `y_px`, `speed_px_sec`, `compartment_stimulus` (60 fps video tracking)
   - Arena chambers: `Novel Mouse` (Social), `Middle`, and `Object` (Non-Social)
