# CHAMP_MIST
A direct replication of the Montreal Imaging Stress Task in PsychoPy.

This implementation has the following features:
- Variable difficulty sum generation using @misty_py
- Event marking via serial port
- Practice screens
- Training sessions for each difficulty to estimate response times
- Trial feedback (Correct, Incorrect, Time-out)
- Block Feedback (Number of correct answers and fallacious response time comparison)
- Slider or keyboard response
- Level-skip (press 's') and testmode (change nreps of testmode loop to 0)

## Event Marking Setup

We have only tested event marking using a Biopac MP36 and BBTK USB-TTL device, and the following seem to work.
The config utility for BBTK is only compatible with Windows.

1. Follow the BBTK instructions to install the USB-TTL module.
2. Connect the BBTK module to the PC and the MP36. The red and green LED's should light up.
3. Open the BBTK USB config tool on the presentation computer and wait a few seconds.
   - If there is no error, click the Apply button.  The Green circle in the Config app should 'light' up - if it does CLOSE THE CONFIG APP.
   - If there is an error check you have set up the device correctly.  If you have, then unplug everything and try again (it usually works after this).
4. On the data collection computer, open the REMSLEEP_CALIBRATION.gtl template in Biopac Student Lab
5. Connect your participant to the MP36 and start recording.
6. Open the serial_check.py script in PsychoPy on the presentation PC; change "COM8" on line 5 to the port that was assigned during set up.
7. Run the script and check that events are being logged on BSL.
  - If they are not, disconnect all wires from the USB-TTL device and repeat 2-6. If they don't after a second attempt, something is wrong with the setup.
  - If they are, then disconnect and reconnect all wires from the USB-TTL device, close BSL without saving data and open the REMSLEEP.gtl template.
9. Click record - recording will not start until a pulse is received from the presentation computer
10. Open CHAMP_MIST.psyexp in PsychoPy, change "COM8" on line 8 of setup_exp code component ion the Welcome Screen block, and click OK.
11. Start the experiment.  If the device is connected, then when you run the study you will get a green dot on the welcome screen.  If it is not, this dot will be red.

## Other Considerations

- You can change the duration of the training session experimental sessions by editing `training.csv` and `experimental_difficulty.csv`.  These are currently set to 60s and 120s respectively.
- Training currently increases in difficulty sequentially, whereas the experimental block randomises between blocks.  This can be changed by editing the `training_trials` and `experimental_trials` loops.
- You can change difficulties by removing or adding rows to `training.csv` and `experimental_difficulty.csv`.


Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
