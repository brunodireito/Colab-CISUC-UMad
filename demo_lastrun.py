#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on julho 15, 2026, at 13:22
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
expName = 'demo'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
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
_fullScr = True
_winSize = (1024, 768)
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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Laboratorio\\Documents\\GitHub\\Colab-CISUC-UMad\\demo_lastrun.py',
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
            logging.getLevel('info')
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
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
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
    
    # --- Initialize components for Routine "Preparation" ---
    instrucao_geral = visual.TextStim(win=win, name='instrucao_geral',
        text='Nesta tarefa, irá observar pequenos vídeos de situações de jogo.\nApós cada vídeo, surgirá uma imagem congelada do lance e deverá tomar uma decisão com base no que vê no ecrã.\nResponda o mais rapidamente e corretamente possível, usando as teclas indicadas (A, B, C ou D).\nEntre ensaios, aparecerá uma cruz de fixação ao centro do ecrã. Durante esse período, mantenha o olhar no centro.\nAntes de começar, haverá um breve período de preparação.\nQuando estiver pronto, prima a barra de espaço para iniciar.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    init_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    # Run 'Begin Experiment' code from lsl_setup
    from pylsl import StreamInfo, StreamOutlet
    
    info = StreamInfo(name='Trigger', type='Markers', channel_count=1, channel_format='int32', source_id='Example')
    outlet = StreamOutlet(info)
    
    # --- Initialize components for Routine "Fixation" ---
    baseline_test = visual.TextStim(win=win, name='baseline_test',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Cues" ---
    # set audio backend
    sound.Sound.backend = 'ptb'
    soundThree = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='soundThree'
    )
    soundThree.setVolume(1.0)
    soundTwo = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='soundTwo'
    )
    soundTwo.setVolume(1.0)
    soundOne = sound.Sound(
        'A', 
        secs=1.0, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='soundOne'
    )
    soundOne.setVolume(1.0)
    
    # --- Initialize components for Routine "fixation_jitter" ---
    fix_trial = visual.TextStim(win=win, name='fix_trial',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Video" ---
    movie = visual.MovieStim(
        win, name='movie',
        filename=None, movieLib='ffpyplayer',
        loop=False, volume=1.0, noAudio=False,
        pos=(0, 0), size=(1.2, 0.9), units=win.units,
        ori=0.0, anchor='center',opacity=1.0, contrast=1.0,
        depth=0
    )
    
    # --- Initialize components for Routine "Decision" ---
    texto_decisao = visual.TextStim(win=win, name='texto_decisao',
        text='Selecione a opção que considera mais adequada para a situação apresentada.\n\nDecisão?\n\nPrima A, B, C ou D\n\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "fixation_post" ---
    fix_cross_post = visual.TextStim(win=win, name='fix_cross_post',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Fixation2_post" ---
    baseline_post = visual.TextStim(win=win, name='baseline_post',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.08, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
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
    
    # --- Prepare to start Routine "Preparation" ---
    # create an object to store info about Routine Preparation
    Preparation = data.Routine(
        name='Preparation',
        components=[instrucao_geral, init_resp],
    )
    Preparation.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for init_resp
    init_resp.keys = []
    init_resp.rt = []
    _init_resp_allKeys = []
    # store start times for Preparation
    Preparation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Preparation.tStart = globalClock.getTime(format='float')
    Preparation.status = STARTED
    thisExp.addData('Preparation.started', Preparation.tStart)
    Preparation.maxDuration = None
    # keep track of which components have finished
    PreparationComponents = Preparation.components
    for thisComponent in Preparation.components:
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
    
    # --- Run Routine "Preparation" ---
    thisExp.currentRoutine = Preparation
    Preparation.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instrucao_geral* updates
        
        # if instrucao_geral is starting this frame...
        if instrucao_geral.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instrucao_geral.frameNStart = frameN  # exact frame index
            instrucao_geral.tStart = t  # local t and not account for scr refresh
            instrucao_geral.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instrucao_geral, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instrucao_geral.started')
            # update status
            instrucao_geral.status = STARTED
            instrucao_geral.setAutoDraw(True)
        
        # if instrucao_geral is active this frame...
        if instrucao_geral.status == STARTED:
            # update params
            pass
        
        # *init_resp* updates
        waitOnFlip = False
        
        # if init_resp is starting this frame...
        if init_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            init_resp.frameNStart = frameN  # exact frame index
            init_resp.tStart = t  # local t and not account for scr refresh
            init_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(init_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'init_resp.started')
            # update status
            init_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(init_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(init_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if init_resp.status == STARTED and not waitOnFlip:
            theseKeys = init_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _init_resp_allKeys.extend(theseKeys)
            if len(_init_resp_allKeys):
                init_resp.keys = _init_resp_allKeys[-1].name  # just the last key pressed
                init_resp.rt = _init_resp_allKeys[-1].rt
                init_resp.duration = _init_resp_allKeys[-1].duration
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
                currentRoutine=Preparation,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Preparation.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Preparation.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Preparation.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Preparation" ---
    for thisComponent in Preparation.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Preparation
    Preparation.tStop = globalClock.getTime(format='float')
    Preparation.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Preparation.stopped', Preparation.tStop)
    # check responses
    if init_resp.keys in ['', [], None]:  # No response was made
        init_resp.keys = None
    thisExp.addData('init_resp.keys',init_resp.keys)
    if init_resp.keys != None:  # we had a response
        thisExp.addData('init_resp.rt', init_resp.rt)
        thisExp.addData('init_resp.duration', init_resp.duration)
    thisExp.nextEntry()
    # the Routine "Preparation" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Fixation" ---
    # create an object to store info about Routine Fixation
    Fixation = data.Routine(
        name='Fixation',
        components=[baseline_test],
    )
    Fixation.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from var_100
    win.callOnFlip(outlet.push_sample, x=[100])
    # store start times for Fixation
    Fixation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Fixation.tStart = globalClock.getTime(format='float')
    Fixation.status = STARTED
    thisExp.addData('Fixation.started', Fixation.tStart)
    Fixation.maxDuration = 25.0
    # keep track of which components have finished
    FixationComponents = Fixation.components
    for thisComponent in Fixation.components:
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
    
    # --- Run Routine "Fixation" ---
    thisExp.currentRoutine = Fixation
    Fixation.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > Fixation.maxDuration-frameTolerance:
            Fixation.maxDurationReached = True
            continueRoutine = False
        
        # *baseline_test* updates
        
        # if baseline_test is starting this frame...
        if baseline_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            baseline_test.frameNStart = frameN  # exact frame index
            baseline_test.tStart = t  # local t and not account for scr refresh
            baseline_test.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(baseline_test, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'baseline_test.started')
            # update status
            baseline_test.status = STARTED
            baseline_test.setAutoDraw(True)
        
        # if baseline_test is active this frame...
        if baseline_test.status == STARTED:
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
                currentRoutine=Fixation,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Fixation.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Fixation.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Fixation.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Fixation" ---
    for thisComponent in Fixation.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Fixation
    Fixation.tStop = globalClock.getTime(format='float')
    Fixation.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Fixation.stopped', Fixation.tStop)
    thisExp.nextEntry()
    # the Routine "Fixation" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Cues" ---
    # create an object to store info about Routine Cues
    Cues = data.Routine(
        name='Cues',
        components=[soundThree, soundTwo, soundOne],
    )
    Cues.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    soundThree.setSound('resources/sounds/3_08_0.wav', secs=1.0, hamming=True)
    soundThree.setVolume(1.0, log=False)
    soundThree.seek(0)
    soundTwo.setSound('resources/sounds/2_08_0.wav', secs=1.0, hamming=True)
    soundTwo.setVolume(1.0, log=False)
    soundTwo.seek(0)
    soundOne.setSound('resources/sounds/1_08_0.wav', secs=1.0, hamming=True)
    soundOne.setVolume(1.0, log=False)
    soundOne.seek(0)
    # store start times for Cues
    Cues.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Cues.tStart = globalClock.getTime(format='float')
    Cues.status = STARTED
    thisExp.addData('Cues.started', Cues.tStart)
    Cues.maxDuration = None
    # keep track of which components have finished
    CuesComponents = Cues.components
    for thisComponent in Cues.components:
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
    
    # --- Run Routine "Cues" ---
    thisExp.currentRoutine = Cues
    Cues.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *soundThree* updates
        
        # if soundThree is starting this frame...
        if soundThree.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            soundThree.frameNStart = frameN  # exact frame index
            soundThree.tStart = t  # local t and not account for scr refresh
            soundThree.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('soundThree.started', tThisFlipGlobal)
            # update status
            soundThree.status = STARTED
            soundThree.play(when=win)  # sync with win flip
        
        # if soundThree is stopping this frame...
        if soundThree.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > soundThree.tStartRefresh + 1.0-frameTolerance or soundThree.isFinished:
                # keep track of stop time/frame for later
                soundThree.tStop = t  # not accounting for scr refresh
                soundThree.tStopRefresh = tThisFlipGlobal  # on global time
                soundThree.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'soundThree.stopped')
                # update status
                soundThree.status = FINISHED
                soundThree.stop()
        
        # *soundTwo* updates
        
        # if soundTwo is starting this frame...
        if soundTwo.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
            # keep track of start time/frame for later
            soundTwo.frameNStart = frameN  # exact frame index
            soundTwo.tStart = t  # local t and not account for scr refresh
            soundTwo.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('soundTwo.started', tThisFlipGlobal)
            # update status
            soundTwo.status = STARTED
            soundTwo.play(when=win)  # sync with win flip
        
        # if soundTwo is stopping this frame...
        if soundTwo.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > soundTwo.tStartRefresh + 1.0-frameTolerance or soundTwo.isFinished:
                # keep track of stop time/frame for later
                soundTwo.tStop = t  # not accounting for scr refresh
                soundTwo.tStopRefresh = tThisFlipGlobal  # on global time
                soundTwo.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'soundTwo.stopped')
                # update status
                soundTwo.status = FINISHED
                soundTwo.stop()
        
        # *soundOne* updates
        
        # if soundOne is starting this frame...
        if soundOne.status == NOT_STARTED and tThisFlip >= 2.0-frameTolerance:
            # keep track of start time/frame for later
            soundOne.frameNStart = frameN  # exact frame index
            soundOne.tStart = t  # local t and not account for scr refresh
            soundOne.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('soundOne.started', tThisFlipGlobal)
            # update status
            soundOne.status = STARTED
            soundOne.play(when=win)  # sync with win flip
        
        # if soundOne is stopping this frame...
        if soundOne.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > soundOne.tStartRefresh + 1.0-frameTolerance or soundOne.isFinished:
                # keep track of stop time/frame for later
                soundOne.tStop = t  # not accounting for scr refresh
                soundOne.tStopRefresh = tThisFlipGlobal  # on global time
                soundOne.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'soundOne.stopped')
                # update status
                soundOne.status = FINISHED
                soundOne.stop()
        
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
                currentRoutine=Cues,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Cues.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Cues.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Cues.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Cues" ---
    for thisComponent in Cues.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Cues
    Cues.tStop = globalClock.getTime(format='float')
    Cues.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Cues.stopped', Cues.tStop)
    soundThree.pause()  # ensure sound has stopped at end of Routine
    soundTwo.pause()  # ensure sound has stopped at end of Routine
    soundOne.pause()  # ensure sound has stopped at end of Routine
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Cues.maxDurationReached:
        routineTimer.addTime(-Cues.maxDuration)
    elif Cues.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('selector.csv'), 
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
        
        # --- Prepare to start Routine "fixation_jitter" ---
        # create an object to store info about Routine fixation_jitter
        fixation_jitter = data.Routine(
            name='fixation_jitter',
            components=[fix_trial],
        )
        fixation_jitter.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        fix_trial.setText('+')
        # Run 'Begin Routine' code from var_110
        win.callOnFlip(outlet.push_sample, x=[110])
        jitter_dur = randint(6, 10)
        # store start times for fixation_jitter
        fixation_jitter.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_jitter.tStart = globalClock.getTime(format='float')
        fixation_jitter.status = STARTED
        thisExp.addData('fixation_jitter.started', fixation_jitter.tStart)
        fixation_jitter.maxDuration = 8.0
        # keep track of which components have finished
        fixation_jitterComponents = fixation_jitter.components
        for thisComponent in fixation_jitter.components:
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
        
        # --- Run Routine "fixation_jitter" ---
        thisExp.currentRoutine = fixation_jitter
        fixation_jitter.forceEnded = routineForceEnded = not continueRoutine
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
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > fixation_jitter.maxDuration-frameTolerance:
                fixation_jitter.maxDurationReached = True
                continueRoutine = False
            
            # *fix_trial* updates
            
            # if fix_trial is starting this frame...
            if fix_trial.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_trial.frameNStart = frameN  # exact frame index
                fix_trial.tStart = t  # local t and not account for scr refresh
                fix_trial.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_trial, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_trial.started')
                # update status
                fix_trial.status = STARTED
                fix_trial.setAutoDraw(True)
            
            # if fix_trial is active this frame...
            if fix_trial.status == STARTED:
                # update params
                pass
            
            # if fix_trial is stopping this frame...
            if fix_trial.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_trial.tStartRefresh + jitter_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_trial.tStop = t  # not accounting for scr refresh
                    fix_trial.tStopRefresh = tThisFlipGlobal  # on global time
                    fix_trial.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_trial.stopped')
                    # update status
                    fix_trial.status = FINISHED
                    fix_trial.setAutoDraw(False)
            
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
                    currentRoutine=fixation_jitter,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_jitter.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_jitter.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_jitter.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_jitter" ---
        for thisComponent in fixation_jitter.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_jitter
        fixation_jitter.tStop = globalClock.getTime(format='float')
        fixation_jitter.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_jitter.stopped', fixation_jitter.tStop)
        # the Routine "fixation_jitter" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Video" ---
        # create an object to store info about Routine Video
        Video = data.Routine(
            name='Video',
            components=[movie],
        )
        Video.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        movie.setMovie(videos)
        # Run 'Begin Routine' code from var_120
        win.callOnFlip(outlet.push_sample, x=[120])
        # store start times for Video
        Video.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Video.tStart = globalClock.getTime(format='float')
        Video.status = STARTED
        thisExp.addData('Video.started', Video.tStart)
        Video.maxDuration = None
        # keep track of which components have finished
        VideoComponents = Video.components
        for thisComponent in Video.components:
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
        
        # --- Run Routine "Video" ---
        thisExp.currentRoutine = Video
        Video.forceEnded = routineForceEnded = not continueRoutine
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
            
            # *movie* updates
            
            # if movie is starting this frame...
            if movie.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                movie.frameNStart = frameN  # exact frame index
                movie.tStart = t  # local t and not account for scr refresh
                movie.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(movie, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'movie.started')
                # update status
                movie.status = STARTED
                movie.setAutoDraw(True)
                movie.play()
            
            # if movie is stopping this frame...
            if movie.status == STARTED:
                if bool(False) or movie.isFinished:
                    # keep track of stop time/frame for later
                    movie.tStop = t  # not accounting for scr refresh
                    movie.tStopRefresh = tThisFlipGlobal  # on global time
                    movie.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'movie.stopped')
                    # update status
                    movie.status = FINISHED
                    movie.setAutoDraw(False)
            
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
                    currentRoutine=Video,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Video.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Video.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Video.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Video" ---
        for thisComponent in Video.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Video
        Video.tStop = globalClock.getTime(format='float')
        Video.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Video.stopped', Video.tStop)
        movie.setAutoDraw(False)
        movie.stop()  # ensure movie has stopped at end of Routine
        # the Routine "Video" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Decision" ---
        # create an object to store info about Routine Decision
        Decision = data.Routine(
            name='Decision',
            components=[texto_decisao, key_resp],
        )
        Decision.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # Run 'Begin Routine' code from var_130
        win.callOnFlip(outlet.push_sample, x=[130])
        response_sent = False
        # store start times for Decision
        Decision.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Decision.tStart = globalClock.getTime(format='float')
        Decision.status = STARTED
        thisExp.addData('Decision.started', Decision.tStart)
        Decision.maxDuration = 6.0
        # keep track of which components have finished
        DecisionComponents = Decision.components
        for thisComponent in Decision.components:
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
        
        # --- Run Routine "Decision" ---
        thisExp.currentRoutine = Decision
        Decision.forceEnded = routineForceEnded = not continueRoutine
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
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > Decision.maxDuration-frameTolerance:
                Decision.maxDurationReached = True
                continueRoutine = False
            
            # *texto_decisao* updates
            
            # if texto_decisao is starting this frame...
            if texto_decisao.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                texto_decisao.frameNStart = frameN  # exact frame index
                texto_decisao.tStart = t  # local t and not account for scr refresh
                texto_decisao.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(texto_decisao, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'texto_decisao.started')
                # update status
                texto_decisao.status = STARTED
                texto_decisao.setAutoDraw(True)
            
            # if texto_decisao is active this frame...
            if texto_decisao.status == STARTED:
                # update params
                pass
            
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
                if tThisFlipGlobal > key_resp.tStartRefresh + 6-frameTolerance:
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
                theseKeys = key_resp.getKeys(keyList=['a', 'b', 'c', 'd'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # was this correct?
                    if (key_resp.keys == str(corrAns)) or (key_resp.keys == corrAns):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from var_130
            if key_resp.keys is not None and not response_sent:
                if key_resp.keys == 'a':
                    outlet.push_sample(x=[140])
                elif key_resp.keys == 'b':
                    outlet.push_sample(x=[141])
                elif key_resp.keys == 'c':
                    outlet.push_sample(x=[142])
                elif key_resp.keys == 'd':
                    outlet.push_sample(x=[143])
                response_sent = True
            
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
                    currentRoutine=Decision,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                Decision.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if Decision.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in Decision.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Decision" ---
        for thisComponent in Decision.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Decision
        Decision.tStop = globalClock.getTime(format='float')
        Decision.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Decision.stopped', Decision.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(corrAns).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('key_resp.keys',key_resp.keys)
        trials.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
            trials.addData('key_resp.duration', key_resp.duration)
        # the Routine "Decision" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "fixation_post" ---
        # create an object to store info about Routine fixation_post
        fixation_post = data.Routine(
            name='fixation_post',
            components=[fix_cross_post],
        )
        fixation_post.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from var_150
        win.callOnFlip(outlet.push_sample, x=[150])
        # store start times for fixation_post
        fixation_post.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_post.tStart = globalClock.getTime(format='float')
        fixation_post.status = STARTED
        thisExp.addData('fixation_post.started', fixation_post.tStart)
        fixation_post.maxDuration = 2.5
        # keep track of which components have finished
        fixation_postComponents = fixation_post.components
        for thisComponent in fixation_post.components:
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
        
        # --- Run Routine "fixation_post" ---
        thisExp.currentRoutine = fixation_post
        fixation_post.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrial, 'status') and thisTrial.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > fixation_post.maxDuration-frameTolerance:
                fixation_post.maxDurationReached = True
                continueRoutine = False
            
            # *fix_cross_post* updates
            
            # if fix_cross_post is starting this frame...
            if fix_cross_post.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_cross_post.frameNStart = frameN  # exact frame index
                fix_cross_post.tStart = t  # local t and not account for scr refresh
                fix_cross_post.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_cross_post, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_cross_post.started')
                # update status
                fix_cross_post.status = STARTED
                fix_cross_post.setAutoDraw(True)
            
            # if fix_cross_post is active this frame...
            if fix_cross_post.status == STARTED:
                # update params
                pass
            
            # if fix_cross_post is stopping this frame...
            if fix_cross_post.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fix_cross_post.tStartRefresh + 2.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fix_cross_post.tStop = t  # not accounting for scr refresh
                    fix_cross_post.tStopRefresh = tThisFlipGlobal  # on global time
                    fix_cross_post.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_cross_post.stopped')
                    # update status
                    fix_cross_post.status = FINISHED
                    fix_cross_post.setAutoDraw(False)
            
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
                    currentRoutine=fixation_post,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_post.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_post.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_post.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_post" ---
        for thisComponent in fixation_post.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_post
        fixation_post.tStop = globalClock.getTime(format='float')
        fixation_post.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_post.stopped', fixation_post.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_post.maxDurationReached:
            routineTimer.addTime(-fixation_post.maxDuration)
        elif fixation_post.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.500000)
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
        
    # completed 1 repeats of 'trials'
    trials.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Fixation2_post" ---
    # create an object to store info about Routine Fixation2_post
    Fixation2_post = data.Routine(
        name='Fixation2_post',
        components=[baseline_post],
    )
    Fixation2_post.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from var_160
    win.callOnFlip(outlet.push_sample, x=[160])
    # store start times for Fixation2_post
    Fixation2_post.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Fixation2_post.tStart = globalClock.getTime(format='float')
    Fixation2_post.status = STARTED
    thisExp.addData('Fixation2_post.started', Fixation2_post.tStart)
    Fixation2_post.maxDuration = 25.0
    # keep track of which components have finished
    Fixation2_postComponents = Fixation2_post.components
    for thisComponent in Fixation2_post.components:
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
    
    # --- Run Routine "Fixation2_post" ---
    thisExp.currentRoutine = Fixation2_post
    Fixation2_post.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > Fixation2_post.maxDuration-frameTolerance:
            Fixation2_post.maxDurationReached = True
            continueRoutine = False
        
        # *baseline_post* updates
        
        # if baseline_post is starting this frame...
        if baseline_post.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            baseline_post.frameNStart = frameN  # exact frame index
            baseline_post.tStart = t  # local t and not account for scr refresh
            baseline_post.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(baseline_post, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'baseline_post.started')
            # update status
            baseline_post.status = STARTED
            baseline_post.setAutoDraw(True)
        
        # if baseline_post is active this frame...
        if baseline_post.status == STARTED:
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
                currentRoutine=Fixation2_post,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            Fixation2_post.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if Fixation2_post.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in Fixation2_post.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Fixation2_post" ---
    for thisComponent in Fixation2_post.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Fixation2_post
    Fixation2_post.tStop = globalClock.getTime(format='float')
    Fixation2_post.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Fixation2_post.stopped', Fixation2_post.tStop)
    thisExp.nextEntry()
    # the Routine "Fixation2_post" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
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
    expInfo = showExpInfoDlg(expInfo=expInfo)
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
