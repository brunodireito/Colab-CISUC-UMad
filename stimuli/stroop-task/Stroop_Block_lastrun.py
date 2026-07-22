#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on julho 22, 2026, at 15:47
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.3'
expName = 'stroop_complex'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': 'XXX',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1600, 900]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data' + os.sep + '%s_%s' % (expInfo['participant'], expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Laboratorio\\Documents\\GitHub\\Colab-CISUC-UMad\\stimuli\\stroop-task\\Stroop_Block_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('warning')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color='black', colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='norm',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = 'black'
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'norm'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Presenter_Ins" ---
    instrText = visual.TextStim(win=win, name='instrText',
        text='Presenter\nPlease Test LSL Connection\n\nPress space to continue',
        font='Arial',
        pos=[0, 0], draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color=[1, 1, 1], colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    ready = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from code_2
    from pylsl import StreamInfo, StreamOutlet # import required classes
    info = StreamInfo(name='Trigger', type='Markers', channel_count=1, channel_format='int32', source_id='Example') # sets variables for object info
    outlet = StreamOutlet(info) # initialize stream.
    
    # --- Initialize components for Routine "Participant_Ins_1" ---
    MoveOn = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Participant:\nOn each screen, Answer:\n \nDo the colors of the top letters\nmatch the meaning of the bottom word?\n    \nq (yes)          p (no)\n\nPress Space to Continue',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.11, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "Participant_Ins_2" ---
    Moveon2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    PI_2 = visual.TextStim(win=win, name='PI_2',
        text='E.g.: Do the colors of the top letters \nmatch the meaning of the bottom word?\n\n',
        font='Arial',
        pos=(0, 0.70), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    text = visual.TextStim(win=win, name='text',
        text='green',
        font='Arial',
        pos=(0.25, 0.4), draggable=False, height=0.2, wrapWidth=None, ori=0, 
        color='blue', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    text_3 = visual.TextStim(win=win, name='text_3',
        text='red',
        font='Arial',
        pos=(-0.25, 0.4), draggable=False, height=0.2, wrapWidth=None, ori=0, 
        color='blue', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    text_4 = visual.TextStim(win=win, name='text_4',
        text='\nred          blue\n\nno           yes',
        font='Arial',
        pos=(0, -0.1), draggable=False, height=0.2, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "Participant_Ins_3" ---
    PI_3 = visual.TextStim(win=win, name='PI_3',
        text='You will now practice 10 times\n\nThe stimuli will last for less than 2 sec\nTry to respond before the screen dissapears\n\nq (yes)    p (no)\n\n\nPress Space to Continue',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    MoveOn3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Prac" ---
    ColorWordP = visual.TextStim(win=win, name='ColorWordP',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.2, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    WhiteWordP = visual.TextStim(win=win, name='WhiteWordP',
        text='',
        font='Arial',
        pos=(0, -0.25), draggable=False, height=0.2, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "feedback" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.15, wrapWidth=None, ori=0, 
        color='orange', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    text_6 = visual.TextStim(win=win, name='text_6',
        text='',
        font='Arial',
        pos=(0, 0.5), draggable=False, height=0.15, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    text_7 = visual.TextStim(win=win, name='text_7',
        text='',
        font='Arial',
        pos=(0, -0.25), draggable=False, height=0.15, wrapWidth=None, ori=0, 
        color='orange', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-2.0);
    text_8 = visual.TextStim(win=win, name='text_8',
        text='Your answer',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "BeginTrials" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text='We will now begin the trials\n\nThey will appear in blocks\n\nStare at + sign in between trial blocks\n\nPress space when ready',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    Moveon4 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Trigger" ---
    
    # --- Initialize components for Routine "trial" ---
    Colorword = visual.TextStim(win=win, name='Colorword',
        text='',
        font='Arial',
        pos=[0, 0.5], draggable=False, height=0.2, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    Whiteword = visual.TextStim(win=win, name='Whiteword',
        text='',
        font='Arial',
        pos=(0, -0.5), draggable=False, height=0.2, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=-1.0);
    resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "Rest_2" ---
    Rest_ = visual.TextStim(win=win, name='Rest_',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.2, wrapWidth=None, ori=0, 
        color='white', colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "thanks" ---
    thanksText = visual.TextStim(win=win, name='thanksText',
        text='This is the end of the experiment.\n\nThanks!',
        font='arial',
        pos=[0, 0], draggable=False, height=0.1, wrapWidth=None, ori=0, 
        color=[1, 1, 1], colorSpace='rgb', opacity=1, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Presenter_Ins" ---
    # create an object to store info about Routine Presenter_Ins
    Presenter_Ins = data.Routine(
        name='Presenter_Ins',
        components=[instrText, ready],
    )
    Presenter_Ins.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for ready
    ready.keys = []
    ready.rt = []
    _ready_allKeys = []
    # store start times for Presenter_Ins
    Presenter_Ins.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Presenter_Ins.tStart = globalClock.getTime(format='float')
    Presenter_Ins.status = STARTED
    thisExp.addData('Presenter_Ins.started', Presenter_Ins.tStart)
    Presenter_Ins.maxDuration = None
    # keep track of which components have finished
    Presenter_InsComponents = Presenter_Ins.components
    for thisComponent in Presenter_Ins.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Presenter_Ins" ---
    thisExp.currentRoutine = Presenter_Ins
    Presenter_Ins.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instrText* updates
        
        # if instrText is starting this frame...
        if instrText.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            instrText.frameNStart = frameN  # exact frame index
            instrText.tStart = t  # local t and not account for scr refresh
            instrText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instrText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instrText.started')
            # update status
            instrText.status = STARTED
            instrText.setAutoDraw(True)
        
        # if instrText is active this frame...
        if instrText.status == STARTED:
            # update params
            pass
        
        # *ready* updates
        waitOnFlip = False
        
        # if ready is starting this frame...
        if ready.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            ready.frameNStart = frameN  # exact frame index
            ready.tStart = t  # local t and not account for scr refresh
            ready.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(ready, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'ready.started')
            # update status
            ready.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(ready.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(ready.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if ready.status == STARTED and not waitOnFlip:
            theseKeys = ready.getKeys(keyList=["space"], ignoreKeys=["escape"], waitRelease=False)
            _ready_allKeys.extend(theseKeys)
            if len(_ready_allKeys):
                ready.keys = _ready_allKeys[-1].name  # just the last key pressed
                ready.rt = _ready_allKeys[-1].rt
                ready.duration = _ready_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Presenter_Ins,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Presenter_Ins.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Presenter_Ins.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Presenter_Ins.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Presenter_Ins" ---
    for thisComponent in Presenter_Ins.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Presenter_Ins
    Presenter_Ins.tStop = globalClock.getTime(format='float')
    Presenter_Ins.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Presenter_Ins.stopped', Presenter_Ins.tStop)
    thisExp.nextEntry()
    # the Routine "Presenter_Ins" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Participant_Ins_1" ---
    # create an object to store info about Routine Participant_Ins_1
    Participant_Ins_1 = data.Routine(
        name='Participant_Ins_1',
        components=[MoveOn, text_2],
    )
    Participant_Ins_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for MoveOn
    MoveOn.keys = []
    MoveOn.rt = []
    _MoveOn_allKeys = []
    # store start times for Participant_Ins_1
    Participant_Ins_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Participant_Ins_1.tStart = globalClock.getTime(format='float')
    Participant_Ins_1.status = STARTED
    thisExp.addData('Participant_Ins_1.started', Participant_Ins_1.tStart)
    Participant_Ins_1.maxDuration = None
    # keep track of which components have finished
    Participant_Ins_1Components = Participant_Ins_1.components
    for thisComponent in Participant_Ins_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Participant_Ins_1" ---
    thisExp.currentRoutine = Participant_Ins_1
    Participant_Ins_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *MoveOn* updates
        waitOnFlip = False
        
        # if MoveOn is starting this frame...
        if MoveOn.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            MoveOn.frameNStart = frameN  # exact frame index
            MoveOn.tStart = t  # local t and not account for scr refresh
            MoveOn.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(MoveOn, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'MoveOn.started')
            # update status
            MoveOn.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(MoveOn.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(MoveOn.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if MoveOn.status == STARTED and not waitOnFlip:
            theseKeys = MoveOn.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _MoveOn_allKeys.extend(theseKeys)
            if len(_MoveOn_allKeys):
                MoveOn.keys = _MoveOn_allKeys[-1].name  # just the last key pressed
                MoveOn.rt = _MoveOn_allKeys[-1].rt
                MoveOn.duration = _MoveOn_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Participant_Ins_1,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Participant_Ins_1.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Participant_Ins_1.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Participant_Ins_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Participant_Ins_1" ---
    for thisComponent in Participant_Ins_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Participant_Ins_1
    Participant_Ins_1.tStop = globalClock.getTime(format='float')
    Participant_Ins_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Participant_Ins_1.stopped', Participant_Ins_1.tStop)
    # check responses
    if MoveOn.keys in ['', [], None]:  # No response was made
        MoveOn.keys = None
    thisExp.addData('MoveOn.keys',MoveOn.keys)
    if MoveOn.keys != None:  # we had a response
        thisExp.addData('MoveOn.rt', MoveOn.rt)
        thisExp.addData('MoveOn.duration', MoveOn.duration)
    thisExp.nextEntry()
    # the Routine "Participant_Ins_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Participant_Ins_2" ---
    # create an object to store info about Routine Participant_Ins_2
    Participant_Ins_2 = data.Routine(
        name='Participant_Ins_2',
        components=[Moveon2, PI_2, text, text_3, text_4],
    )
    Participant_Ins_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Moveon2
    Moveon2.keys = []
    Moveon2.rt = []
    _Moveon2_allKeys = []
    # store start times for Participant_Ins_2
    Participant_Ins_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Participant_Ins_2.tStart = globalClock.getTime(format='float')
    Participant_Ins_2.status = STARTED
    thisExp.addData('Participant_Ins_2.started', Participant_Ins_2.tStart)
    Participant_Ins_2.maxDuration = None
    # keep track of which components have finished
    Participant_Ins_2Components = Participant_Ins_2.components
    for thisComponent in Participant_Ins_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Participant_Ins_2" ---
    thisExp.currentRoutine = Participant_Ins_2
    Participant_Ins_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Moveon2* updates
        waitOnFlip = False
        
        # if Moveon2 is starting this frame...
        if Moveon2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Moveon2.frameNStart = frameN  # exact frame index
            Moveon2.tStart = t  # local t and not account for scr refresh
            Moveon2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Moveon2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Moveon2.started')
            # update status
            Moveon2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Moveon2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Moveon2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Moveon2.status == STARTED and not waitOnFlip:
            theseKeys = Moveon2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Moveon2_allKeys.extend(theseKeys)
            if len(_Moveon2_allKeys):
                Moveon2.keys = _Moveon2_allKeys[-1].name  # just the last key pressed
                Moveon2.rt = _Moveon2_allKeys[-1].rt
                Moveon2.duration = _Moveon2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *PI_2* updates
        
        # if PI_2 is starting this frame...
        if PI_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            PI_2.frameNStart = frameN  # exact frame index
            PI_2.tStart = t  # local t and not account for scr refresh
            PI_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(PI_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'PI_2.started')
            # update status
            PI_2.status = STARTED
            PI_2.setAutoDraw(True)
        
        # if PI_2 is active this frame...
        if PI_2.status == STARTED:
            # update params
            pass
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Participant_Ins_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Participant_Ins_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Participant_Ins_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Participant_Ins_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Participant_Ins_2" ---
    for thisComponent in Participant_Ins_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Participant_Ins_2
    Participant_Ins_2.tStop = globalClock.getTime(format='float')
    Participant_Ins_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Participant_Ins_2.stopped', Participant_Ins_2.tStop)
    # check responses
    if Moveon2.keys in ['', [], None]:  # No response was made
        Moveon2.keys = None
    thisExp.addData('Moveon2.keys',Moveon2.keys)
    if Moveon2.keys != None:  # we had a response
        thisExp.addData('Moveon2.rt', Moveon2.rt)
        thisExp.addData('Moveon2.duration', Moveon2.duration)
    thisExp.nextEntry()
    # the Routine "Participant_Ins_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Participant_Ins_3" ---
    # create an object to store info about Routine Participant_Ins_3
    Participant_Ins_3 = data.Routine(
        name='Participant_Ins_3',
        components=[PI_3, MoveOn3],
    )
    Participant_Ins_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for MoveOn3
    MoveOn3.keys = []
    MoveOn3.rt = []
    _MoveOn3_allKeys = []
    # store start times for Participant_Ins_3
    Participant_Ins_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Participant_Ins_3.tStart = globalClock.getTime(format='float')
    Participant_Ins_3.status = STARTED
    thisExp.addData('Participant_Ins_3.started', Participant_Ins_3.tStart)
    Participant_Ins_3.maxDuration = None
    # keep track of which components have finished
    Participant_Ins_3Components = Participant_Ins_3.components
    for thisComponent in Participant_Ins_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Participant_Ins_3" ---
    thisExp.currentRoutine = Participant_Ins_3
    Participant_Ins_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *PI_3* updates
        
        # if PI_3 is starting this frame...
        if PI_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            PI_3.frameNStart = frameN  # exact frame index
            PI_3.tStart = t  # local t and not account for scr refresh
            PI_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(PI_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'PI_3.started')
            # update status
            PI_3.status = STARTED
            PI_3.setAutoDraw(True)
        
        # if PI_3 is active this frame...
        if PI_3.status == STARTED:
            # update params
            pass
        
        # *MoveOn3* updates
        waitOnFlip = False
        
        # if MoveOn3 is starting this frame...
        if MoveOn3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            MoveOn3.frameNStart = frameN  # exact frame index
            MoveOn3.tStart = t  # local t and not account for scr refresh
            MoveOn3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(MoveOn3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'MoveOn3.started')
            # update status
            MoveOn3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(MoveOn3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(MoveOn3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if MoveOn3.status == STARTED and not waitOnFlip:
            theseKeys = MoveOn3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _MoveOn3_allKeys.extend(theseKeys)
            if len(_MoveOn3_allKeys):
                MoveOn3.keys = _MoveOn3_allKeys[-1].name  # just the last key pressed
                MoveOn3.rt = _MoveOn3_allKeys[-1].rt
                MoveOn3.duration = _MoveOn3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Participant_Ins_3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Participant_Ins_3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Participant_Ins_3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Participant_Ins_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Participant_Ins_3" ---
    for thisComponent in Participant_Ins_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Participant_Ins_3
    Participant_Ins_3.tStop = globalClock.getTime(format='float')
    Participant_Ins_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Participant_Ins_3.stopped', Participant_Ins_3.tStop)
    # check responses
    if MoveOn3.keys in ['', [], None]:  # No response was made
        MoveOn3.keys = None
    thisExp.addData('MoveOn3.keys',MoveOn3.keys)
    if MoveOn3.keys != None:  # we had a response
        thisExp.addData('MoveOn3.rt', MoveOn3.rt)
        thisExp.addData('MoveOn3.duration', MoveOn3.duration)
    thisExp.nextEntry()
    # the Routine "Participant_Ins_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    Practice = data.TrialHandler2(
        name='Practice',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('TrialTypes_All.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(Practice)  # add the loop to the experiment
    thisPractice = Practice.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
    if thisPractice != None:
        for paramName in thisPractice:
            globals()[paramName] = thisPractice[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice in Practice:
        Practice.status = STARTED
        if hasattr(thisPractice, 'status'):
            thisPractice.status = STARTED
        currentLoop = Practice
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice.rgb)
        if thisPractice != None:
            for paramName in thisPractice:
                globals()[paramName] = thisPractice[paramName]
        
        # --- Prepare to start Routine "Prac" ---
        # create an object to store info about Routine Prac
        Prac = data.Routine(
            name='Prac',
            components=[ColorWordP, WhiteWordP, key_resp],
        )
        Prac.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        ColorWordP.setColor(letterColor, colorSpace='rgb')
        ColorWordP.setText(colorword)
        WhiteWordP.setText(blackword)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for Prac
        Prac.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Prac.tStart = globalClock.getTime(format='float')
        Prac.status = STARTED
        thisExp.addData('Prac.started', Prac.tStart)
        Prac.maxDuration = None
        # keep track of which components have finished
        PracComponents = Prac.components
        for thisComponent in Prac.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Prac" ---
        thisExp.currentRoutine = Prac
        Prac.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.75:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *ColorWordP* updates
            
            # if ColorWordP is starting this frame...
            if ColorWordP.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ColorWordP.frameNStart = frameN  # exact frame index
                ColorWordP.tStart = t  # local t and not account for scr refresh
                ColorWordP.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ColorWordP, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'ColorWordP.started')
                # update status
                ColorWordP.status = STARTED
                ColorWordP.setAutoDraw(True)
            
            # if ColorWordP is active this frame...
            if ColorWordP.status == STARTED:
                # update params
                pass
            
            # if ColorWordP is stopping this frame...
            if ColorWordP.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ColorWordP.tStartRefresh + 1.75-frameTolerance:
                    # keep track of stop time/frame for later
                    ColorWordP.tStop = t  # not accounting for scr refresh
                    ColorWordP.tStopRefresh = tThisFlipGlobal  # on global time
                    ColorWordP.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'ColorWordP.stopped')
                    # update status
                    ColorWordP.status = FINISHED
                    ColorWordP.setAutoDraw(False)
            
            # *WhiteWordP* updates
            
            # if WhiteWordP is starting this frame...
            if WhiteWordP.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                WhiteWordP.frameNStart = frameN  # exact frame index
                WhiteWordP.tStart = t  # local t and not account for scr refresh
                WhiteWordP.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(WhiteWordP, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'WhiteWordP.started')
                # update status
                WhiteWordP.status = STARTED
                WhiteWordP.setAutoDraw(True)
            
            # if WhiteWordP is active this frame...
            if WhiteWordP.status == STARTED:
                # update params
                pass
            
            # if WhiteWordP is stopping this frame...
            if WhiteWordP.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > WhiteWordP.tStartRefresh + 1.75-frameTolerance:
                    # keep track of stop time/frame for later
                    WhiteWordP.tStop = t  # not accounting for scr refresh
                    WhiteWordP.tStopRefresh = tThisFlipGlobal  # on global time
                    WhiteWordP.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'WhiteWordP.stopped')
                    # update status
                    WhiteWordP.status = FINISHED
                    WhiteWordP.setAutoDraw(False)
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 1.75-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['q','p'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Prac,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Prac.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Prac.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Prac.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Prac" ---
        for thisComponent in Prac.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Prac
        Prac.tStop = globalClock.getTime(format='float')
        Prac.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Prac.stopped', Prac.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        Practice.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            Practice.addData('key_resp.rt', key_resp.rt)
            Practice.addData('key_resp.duration', key_resp.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Prac.maxDurationReached:
            routineTimer.addTime(-Prac.maxDuration)
        elif Prac.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.750000)
        
        # --- Prepare to start Routine "feedback" ---
        # create an object to store info about Routine feedback
        feedback = data.Routine(
            name='feedback',
            components=[text_5, text_6, text_7, text_8],
        )
        feedback.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_5.setText(corrAns
        )
        text_6.setText('Correct answer')
        text_7.setText(key_resp.keys
        )
        # store start times for feedback
        feedback.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback.tStart = globalClock.getTime(format='float')
        feedback.status = STARTED
        thisExp.addData('feedback.started', feedback.tStart)
        feedback.maxDuration = None
        # keep track of which components have finished
        feedbackComponents = feedback.components
        for thisComponent in feedback.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback" ---
        thisExp.currentRoutine = feedback
        feedback.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice, 'status') and thisPractice.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_5* updates
            
            # if text_5 is starting this frame...
            if text_5.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_5.frameNStart = frameN  # exact frame index
                text_5.tStart = t  # local t and not account for scr refresh
                text_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_5.started')
                # update status
                text_5.status = STARTED
                text_5.setAutoDraw(True)
            
            # if text_5 is active this frame...
            if text_5.status == STARTED:
                # update params
                pass
            
            # if text_5 is stopping this frame...
            if text_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_5.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_5.tStop = t  # not accounting for scr refresh
                    text_5.tStopRefresh = tThisFlipGlobal  # on global time
                    text_5.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_5.stopped')
                    # update status
                    text_5.status = FINISHED
                    text_5.setAutoDraw(False)
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_6.started')
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # if text_6 is stopping this frame...
            if text_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_6.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_6.tStop = t  # not accounting for scr refresh
                    text_6.tStopRefresh = tThisFlipGlobal  # on global time
                    text_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_6.stopped')
                    # update status
                    text_6.status = FINISHED
                    text_6.setAutoDraw(False)
            
            # *text_7* updates
            
            # if text_7 is starting this frame...
            if text_7.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_7.frameNStart = frameN  # exact frame index
                text_7.tStart = t  # local t and not account for scr refresh
                text_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_7.started')
                # update status
                text_7.status = STARTED
                text_7.setAutoDraw(True)
            
            # if text_7 is active this frame...
            if text_7.status == STARTED:
                # update params
                pass
            
            # if text_7 is stopping this frame...
            if text_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_7.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_7.tStop = t  # not accounting for scr refresh
                    text_7.tStopRefresh = tThisFlipGlobal  # on global time
                    text_7.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_7.stopped')
                    # update status
                    text_7.status = FINISHED
                    text_7.setAutoDraw(False)
            
            # *text_8* updates
            
            # if text_8 is starting this frame...
            if text_8.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_8.frameNStart = frameN  # exact frame index
                text_8.tStart = t  # local t and not account for scr refresh
                text_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_8.started')
                # update status
                text_8.status = STARTED
                text_8.setAutoDraw(True)
            
            # if text_8 is active this frame...
            if text_8.status == STARTED:
                # update params
                pass
            
            # if text_8 is stopping this frame...
            if text_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_8.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_8.tStop = t  # not accounting for scr refresh
                    text_8.tStopRefresh = tThisFlipGlobal  # on global time
                    text_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.stopped')
                    # update status
                    text_8.status = FINISHED
                    text_8.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=feedback,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                feedback.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if feedback.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in feedback.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback" ---
        for thisComponent in feedback.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback
        feedback.tStop = globalClock.getTime(format='float')
        feedback.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback.stopped', feedback.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback.maxDurationReached:
            routineTimer.addTime(-feedback.maxDuration)
        elif feedback.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        # mark thisPractice as finished
        if hasattr(thisPractice, 'status'):
            thisPractice.status = FINISHED
        # if awaiting a pause, pause now
        if Practice.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            Practice.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'Practice'
    Practice.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if Practice.trialList in ([], [None], None):
        params = []
    else:
        params = Practice.trialList[0].keys()
    # save data for this loop
    Practice.saveAsExcel(filename + '.xlsx', sheetName='Practice',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "BeginTrials" ---
    # create an object to store info about Routine BeginTrials
    BeginTrials = data.Routine(
        name='BeginTrials',
        components=[text_9, Moveon4],
    )
    BeginTrials.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for Moveon4
    Moveon4.keys = []
    Moveon4.rt = []
    _Moveon4_allKeys = []
    # store start times for BeginTrials
    BeginTrials.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    BeginTrials.tStart = globalClock.getTime(format='float')
    BeginTrials.status = STARTED
    thisExp.addData('BeginTrials.started', BeginTrials.tStart)
    BeginTrials.maxDuration = None
    # keep track of which components have finished
    BeginTrialsComponents = BeginTrials.components
    for thisComponent in BeginTrials.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "BeginTrials" ---
    thisExp.currentRoutine = BeginTrials
    BeginTrials.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_9* updates
        
        # if text_9 is starting this frame...
        if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_9.frameNStart = frameN  # exact frame index
            text_9.tStart = t  # local t and not account for scr refresh
            text_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_9.started')
            # update status
            text_9.status = STARTED
            text_9.setAutoDraw(True)
        
        # if text_9 is active this frame...
        if text_9.status == STARTED:
            # update params
            pass
        
        # *Moveon4* updates
        waitOnFlip = False
        
        # if Moveon4 is starting this frame...
        if Moveon4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Moveon4.frameNStart = frameN  # exact frame index
            Moveon4.tStart = t  # local t and not account for scr refresh
            Moveon4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Moveon4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Moveon4.started')
            # update status
            Moveon4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(Moveon4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(Moveon4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if Moveon4.status == STARTED and not waitOnFlip:
            theseKeys = Moveon4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _Moveon4_allKeys.extend(theseKeys)
            if len(_Moveon4_allKeys):
                Moveon4.keys = _Moveon4_allKeys[-1].name  # just the last key pressed
                Moveon4.rt = _Moveon4_allKeys[-1].rt
                Moveon4.duration = _Moveon4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=BeginTrials,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            BeginTrials.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if BeginTrials.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in BeginTrials.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "BeginTrials" ---
    for thisComponent in BeginTrials.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for BeginTrials
    BeginTrials.tStop = globalClock.getTime(format='float')
    BeginTrials.tStopRefresh = tThisFlipGlobal
    thisExp.addData('BeginTrials.stopped', BeginTrials.tStop)
    # check responses
    if Moveon4.keys in ['', [], None]:  # No response was made
        Moveon4.keys = None
    thisExp.addData('Moveon4.keys',Moveon4.keys)
    if Moveon4.keys != None:  # we had a response
        thisExp.addData('Moveon4.rt', Moveon4.rt)
        thisExp.addData('Moveon4.duration', Moveon4.duration)
    thisExp.nextEntry()
    # the Routine "BeginTrials" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=5.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('ConditionFile.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        trials.status = STARTED
        if hasattr(thisTrial, 'status'):
            thisTrial.status = STARTED
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "Trigger" ---
        # create an object to store info about Routine Trigger
        Trigger = data.Routine(
            name='Trigger',
            components=[],
        )
        Trigger.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        outlet.push_sample(x=[congruent])
        # store start times for Trigger
        Trigger.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Trigger.tStart = globalClock.getTime(format='float')
        Trigger.status = STARTED
        thisExp.addData('Trigger.started', Trigger.tStart)
        Trigger.maxDuration = None
        # keep track of which components have finished
        TriggerComponents = Trigger.components
        for thisComponent in Trigger.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Trigger" ---
        thisExp.currentRoutine = Trigger
        Trigger.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Trigger,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Trigger.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Trigger.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Trigger.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Trigger" ---
        for thisComponent in Trigger.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Trigger
        Trigger.tStop = globalClock.getTime(format='float')
        Trigger.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Trigger.stopped', Trigger.tStop)
        # the Routine "Trigger" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials_Loop = data.TrialHandler2(
            name='trials_Loop',
            nReps=1, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(Condition_File), 
            seed=None, 
            isTrials=True, 
        )
        thisExp.addLoop(trials_Loop)  # add the loop to the experiment
        thisTrials_Loop = trials_Loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_Loop.rgb)
        if thisTrials_Loop != None:
            for paramName in thisTrials_Loop:
                globals()[paramName] = thisTrials_Loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrials_Loop in trials_Loop:
            trials_Loop.status = STARTED
            if hasattr(thisTrials_Loop, 'status'):
                thisTrials_Loop.status = STARTED
            currentLoop = trials_Loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrials_Loop.rgb)
            if thisTrials_Loop != None:
                for paramName in thisTrials_Loop:
                    globals()[paramName] = thisTrials_Loop[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[Colorword, Whiteword, resp],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            Colorword.setColor(letterColor, colorSpace='rgb')
            Colorword.setText(colorword)
            Whiteword.setText(blackword)
            # create starting attributes for resp
            resp.keys = []
            resp.rt = []
            _resp_allKeys = []
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            thisExp.currentRoutine = trial
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.75:
                # if trial has changed, end Routine now
                if hasattr(thisTrials_Loop, 'status') and thisTrials_Loop.status == STOPPING:
                    continueRoutine = False
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Colorword* updates
                
                # if Colorword is starting this frame...
                if Colorword.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    Colorword.frameNStart = frameN  # exact frame index
                    Colorword.tStart = t  # local t and not account for scr refresh
                    Colorword.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Colorword, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Colorword.started')
                    # update status
                    Colorword.status = STARTED
                    Colorword.setAutoDraw(True)
                
                # if Colorword is active this frame...
                if Colorword.status == STARTED:
                    # update params
                    pass
                
                # if Colorword is stopping this frame...
                if Colorword.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Colorword.tStartRefresh + 1.75-frameTolerance:
                        # keep track of stop time/frame for later
                        Colorword.tStop = t  # not accounting for scr refresh
                        Colorword.tStopRefresh = tThisFlipGlobal  # on global time
                        Colorword.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Colorword.stopped')
                        # update status
                        Colorword.status = FINISHED
                        Colorword.setAutoDraw(False)
                
                # *Whiteword* updates
                
                # if Whiteword is starting this frame...
                if Whiteword.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    Whiteword.frameNStart = frameN  # exact frame index
                    Whiteword.tStart = t  # local t and not account for scr refresh
                    Whiteword.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Whiteword, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Whiteword.started')
                    # update status
                    Whiteword.status = STARTED
                    Whiteword.setAutoDraw(True)
                
                # if Whiteword is active this frame...
                if Whiteword.status == STARTED:
                    # update params
                    pass
                
                # if Whiteword is stopping this frame...
                if Whiteword.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Whiteword.tStartRefresh + 1.75-frameTolerance:
                        # keep track of stop time/frame for later
                        Whiteword.tStop = t  # not accounting for scr refresh
                        Whiteword.tStopRefresh = tThisFlipGlobal  # on global time
                        Whiteword.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Whiteword.stopped')
                        # update status
                        Whiteword.status = FINISHED
                        Whiteword.setAutoDraw(False)
                
                # *resp* updates
                waitOnFlip = False
                
                # if resp is starting this frame...
                if resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    resp.frameNStart = frameN  # exact frame index
                    resp.tStart = t  # local t and not account for scr refresh
                    resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'resp.started')
                    # update status
                    resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if resp is stopping this frame...
                if resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > resp.tStartRefresh + 1.75-frameTolerance:
                        # keep track of stop time/frame for later
                        resp.tStop = t  # not accounting for scr refresh
                        resp.tStopRefresh = tThisFlipGlobal  # on global time
                        resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'resp.stopped')
                        # update status
                        resp.status = FINISHED
                        resp.status = FINISHED
                if resp.status == STARTED and not waitOnFlip:
                    theseKeys = resp.getKeys(keyList=['q','p'], ignoreKeys=["escape"], waitRelease=False)
                    _resp_allKeys.extend(theseKeys)
                    if len(_resp_allKeys):
                        resp.keys = _resp_allKeys[-1].name  # just the last key pressed
                        resp.rt = _resp_allKeys[-1].rt
                        resp.duration = _resp_allKeys[-1].duration
                        # was this correct?
                        if (resp.keys == str(corrAns)) or (resp.keys == corrAns):
                            resp.corr = 1
                        else:
                            resp.corr = 0
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer, globalClock], 
                        currentRoutine=trial,
                    )
                    # skip the frame we paused on
                    continue
                
                # has a Component requested the Routine to end?
                if not continueRoutine:
                    trial.forceEnded = routineForceEnded = True
                # has the Routine been forcibly ended?
                if trial.forceEnded or routineForceEnded:
                    break
                # has every Component finished?
                continueRoutine = False
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # check responses
            if resp.keys in ['', [], None]:  # No response was made
                resp.keys = None
                # was no response the correct answer?!
                if str(corrAns).lower() == 'none':
                   resp.corr = 1;  # correct non-response
                else:
                   resp.corr = 0;  # failed to respond (incorrectly)
            # store data for trials_Loop (TrialHandler)
            trials_Loop.addData('resp.keys',resp.keys)
            trials_Loop.addData('resp.corr', resp.corr)
            if resp.keys != None:  # we had a response
                trials_Loop.addData('resp.rt', resp.rt)
                trials_Loop.addData('resp.duration', resp.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.750000)
            # mark thisTrials_Loop as finished
            if hasattr(thisTrials_Loop, 'status'):
                thisTrials_Loop.status = FINISHED
            # if awaiting a pause, pause now
            if trials_Loop.status == PAUSED:
                thisExp.status = PAUSED
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[globalClock], 
                )
                # once done pausing, restore running status
                trials_Loop.status = STARTED
            thisExp.nextEntry()
            
        # completed 1 repeats of 'trials_Loop'
        trials_Loop.status = FINISHED
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if trials_Loop.trialList in ([], [None], None):
            params = []
        else:
            params = trials_Loop.trialList[0].keys()
        # save data for this loop
        trials_Loop.saveAsExcel(filename + '.xlsx', sheetName='trials_Loop',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "Rest_2" ---
        # create an object to store info about Routine Rest_2
        Rest_2 = data.Routine(
            name='Rest_2',
            components=[Rest_],
        )
        Rest_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for Rest_2
        Rest_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Rest_2.tStart = globalClock.getTime(format='float')
        Rest_2.status = STARTED
        thisExp.addData('Rest_2.started', Rest_2.tStart)
        Rest_2.maxDuration = None
        # keep track of which components have finished
        Rest_2Components = Rest_2.components
        for thisComponent in Rest_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Rest_2" ---
        thisExp.currentRoutine = Rest_2
        Rest_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 15.0:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *Rest_* updates
            
            # if Rest_ is starting this frame...
            if Rest_.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                Rest_.frameNStart = frameN  # exact frame index
                Rest_.tStart = t  # local t and not account for scr refresh
                Rest_.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(Rest_, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'Rest_.started')
                # update status
                Rest_.status = STARTED
                Rest_.setAutoDraw(True)
            
            # if Rest_ is active this frame...
            if Rest_.status == STARTED:
                # update params
                pass
            
            # if Rest_ is stopping this frame...
            if Rest_.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > Rest_.tStartRefresh + 15-frameTolerance:
                    # keep track of stop time/frame for later
                    Rest_.tStop = t  # not accounting for scr refresh
                    Rest_.tStopRefresh = tThisFlipGlobal  # on global time
                    Rest_.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Rest_.stopped')
                    # update status
                    Rest_.status = FINISHED
                    Rest_.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=Rest_2,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Rest_2.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Rest_2.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Rest_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Rest_2" ---
        for thisComponent in Rest_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Rest_2
        Rest_2.tStop = globalClock.getTime(format='float')
        Rest_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Rest_2.stopped', Rest_2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Rest_2.maxDurationReached:
            routineTimer.addTime(-Rest_2.maxDuration)
        elif Rest_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-15.000000)
        # mark thisTrial as finished
        if hasattr(thisTrial, 'status'):
            thisTrial.status = FINISHED
        # if awaiting a pause, pause now
        if trials.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trials.status = STARTED
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials.trialList in ([], [None], None):
        params = []
    else:
        params = trials.trialList[0].keys()
    # save data for this loop
    trials.saveAsExcel(filename + '.xlsx', sheetName='trials',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "thanks" ---
    # create an object to store info about Routine thanks
    thanks = data.Routine(
        name='thanks',
        components=[thanksText],
    )
    thanks.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for thanks
    thanks.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thanks.tStart = globalClock.getTime(format='float')
    thanks.status = STARTED
    thisExp.addData('thanks.started', thanks.tStart)
    thanks.maxDuration = None
    # keep track of which components have finished
    thanksComponents = thanks.components
    for thisComponent in thanks.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thanks" ---
    thisExp.currentRoutine = thanks
    thanks.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thanksText* updates
        
        # if thanksText is starting this frame...
        if thanksText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thanksText.frameNStart = frameN  # exact frame index
            thanksText.tStart = t  # local t and not account for scr refresh
            thanksText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thanksText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thanksText.started')
            # update status
            thanksText.status = STARTED
            thanksText.setAutoDraw(True)
        
        # if thanksText is active this frame...
        if thanksText.status == STARTED:
            # update params
            pass
        
        # if thanksText is stopping this frame...
        if thanksText.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thanksText.tStartRefresh + 2.0-frameTolerance:
                # keep track of stop time/frame for later
                thanksText.tStop = t  # not accounting for scr refresh
                thanksText.tStopRefresh = tThisFlipGlobal  # on global time
                thanksText.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thanksText.stopped')
                # update status
                thanksText.status = FINISHED
                thanksText.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=thanks,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            thanks.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if thanks.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in thanks.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thanks" ---
    for thisComponent in thanks.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thanks
    thanks.tStop = globalClock.getTime(format='float')
    thanks.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thanks.stopped', thanks.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thanks.maxDurationReached:
        routineTimer.addTime(-thanks.maxDuration)
    elif thanks.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
