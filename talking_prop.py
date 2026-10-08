#!/usr/bin/env python3
"""talking_prop.py - live voice conversation for an animatronic prop with a moving jaw.

Mic -> OpenAI-style realtime speech-to-speech server (/v1/realtime) -> speaker,
with a jaw servo driven by the loudness of the reply audio, using the same
method and config.ini format as ChatterPi (https://github.com/ViennaMike/ChatterPi).

Files read from the script's own directory:
  config.ini     Servo calibration, levels and pins, in ChatterPi's format. Optional.
  persona.txt    The personality / instructions sent to the model. Optional.
  brave_key.txt  Brave Search API key. Optional; enables web search.
                 (The BRAVE_API_KEY environment variable works too.)

Run "python3 talking_prop.py --help" for options.
"""
import argparse
import base64
import configparser
import datetime
import html
import json
import os
import queue
import re
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

SERVER_RATE = 16000  # the s2s server speaks 16 kHz mono 16-bit PCM both ways
HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_PERSONA = (
    "You are a friendly animatronic character. Keep every reply to one or two short "
    "spoken sentences. Never use lists, emoji or stage directions."
)


# --------------------------------------------------------------------------
# Settings carried over from ChatterPi's config.ini
# --------------------------------------------------------------------------
def load_chatterpi_config(path):
    """Return jaw settings, using ChatterPi's defaults for anything missing."""
    cfg = configparser.ConfigParser()
    cfg.read(path)

    def get(section, key, default):
        try:
            return cfg[section][key]
        except KeyError:
            return default

    return {
        "servo_min": int(get("SERVO", "servo_min", 1050)),
        "servo_max": int(get("SERVO", "servo_max", 1250)),
        "min_angle": int(get("SERVO", "min_angle", 0)),
        "max_angle": int(get("SERVO", "max_angle", 90)),
        "style": int(get("CONTROLLER", "style", 1)),
        "threshold": int(get("CONTROLLER", "threshold", 2000)),
        "level1": int(get("CONTROLLER", "level1", 1500)),
        "level2": int(get("CONTROLLER", "level2", 2500)),
        "level3": int(get("CONTROLLER", "level3", 3500)),
        "eyes": str(get("PROP", "eyes", "OFF")).upper() == "ON",
        "jaw_pin": int(get("PINS", "jaw_pin", 18)),
        "eyes_pin": int(get("PINS", "eyes_pin", 25)),
    }


# --------------------------------------------------------------------------
# Web search (Brave Search API)
# --------------------------------------------------------------------------
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": (
        "Search the web for current or factual information: news, weather, sports scores, "
        "events, opening hours, or anything you are not sure about. Call it immediately instead of "
        "guessing, and never say you will look something up without calling it in the same turn. "
        "Read the results yourself and answer in one or two short sentences, in character. "
        "Never read out web addresses."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "What to search for."}},
        "required": ["query"],
    },
}


TIME_TOOL = {
    "type": "function",
    "name": "get_current_time",
    "description": (
        "Get the current local date and time. Call it whenever you are asked the time, the date, "
        "the day of the week, or how long until something. Never guess the time."
    ),
    "parameters": {"type": "object", "properties": {}},
}

KID_SAFE_RULE = (
    "You are speaking with children and families. Keep everything friendly and age-appropriate. "
    "Never describe crime, death, violence, war, disasters or anything frightening from real life; "
    "if a question or a search result goes there, cheerfully change the subject."
)


# Replies where the model says it will look something up. Smaller models often say this
# without calling web_search; when they do, the client runs the search itself.
PROMISE_RE = re.compile(
    r"\b(i'?ll|i will|let me|lemme|gonna|going to|i'?m off to|i be)\b[^.!?]{0,50}?"
    r"\b(check|look|search|scour|find out|see what|dig|seek|hunt|ask)"
    r"|\b(hold fast|hold on|one moment|a tick|bear with me)\b",
    re.IGNORECASE,
)


def promised_to_search(reply):
    return bool(PROMISE_RE.search(reply or ""))


def current_time():
    now = datetime.datetime.now().astimezone()
    return {
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M %p").lstrip("0"),
        "timezone": now.tzname() or "",
    }


def load_brave_key(path):
    key = os.getenv("BRAVE_API_KEY", "").strip()
    if not key and os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            key = handle.read().strip()
    return key


def _plain(text):
    """Brave marks matches with <strong>; strip tags and HTML entities."""
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def brave_search(key, query, count=5, opener=urllib.request.urlopen):
    """Return a short, model-friendly list of results, or {"error": ...}."""
    url = BRAVE_URL + "?" + urllib.parse.urlencode({"q": query, "count": count, "safesearch": "strict"})
    request = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "X-Subscription-Token": key,
    })
    try:
        with opener(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"error": "search failed: %s" % exc}
    results = []
    for item in (data.get("web") or {}).get("results", [])[:count]:
        results.append({
            "title": _plain(item.get("title")),
            "snippet": _plain(item.get("description")),
            "source": urllib.parse.urlsplit(item.get("url", "")).netloc,
        })
    return {"query": query, "results": results} if results else {"query": query, "results": [], "note": "no results"}


# --------------------------------------------------------------------------
# Jaw
# --------------------------------------------------------------------------
class Jaw:
    """Maps the loudness of a chunk of audio to a servo angle (ChatterPi's method)."""

    def __init__(self, settings, enabled=True):
        self.s = settings
        # Same convention as ChatterPi: the larger angle is the resting position.
        self.closed = max(settings["min_angle"], settings["max_angle"])
        self.open = min(settings["min_angle"], settings["max_angle"])
        self.servo = None
        self.eyes = None
        self.last_angle = None
        if not enabled:
            return
        from gpiozero import AngularServo, DigitalOutputDevice

        factory = None
        try:
            from gpiozero.pins.pigpio import PiGPIOFactory

            factory = PiGPIOFactory()  # hardware-timed pulses: no servo jitter
        except Exception as exc:  # pigpiod not running or not installed
            print("pigpio unavailable (%s); using gpiozero's default pins. "
                  "Expect some servo jitter; see the pigpio section of the README." % exc)
        self.servo = AngularServo(
            settings["jaw_pin"],
            min_angle=settings["min_angle"],
            max_angle=settings["max_angle"],
            initial_angle=None,
            min_pulse_width=settings["servo_min"] / 1e6,
            max_pulse_width=settings["servo_max"] / 1e6,
            pin_factory=factory,
        )
        if settings["eyes"]:
            self.eyes = DigitalOutputDevice(settings["eyes_pin"], pin_factory=factory)

    def target(self, samples):
        """Return the jaw angle for one chunk of int16 samples."""
        if len(samples) == 0:
            return self.closed
        volume = int(np.abs(samples.astype(np.int32)).mean())
        s = self.s
        if s["style"] == 0:  # single threshold: open or shut
            return self.open if volume > s["threshold"] else self.closed
        step = (self.open - self.closed) / 3.0
        if volume > s["level3"]:
            return self.open
        if volume > s["level2"]:
            return self.closed + 2 * step
        if volume > s["level1"]:
            return self.closed + step
        return self.closed

    def move(self, angle):
        if angle == self.last_angle:
            return
        self.last_angle = angle
        if self.servo is not None:
            self.servo.angle = angle

    def speaking(self, on):
        if self.eyes is not None:
            self.eyes.on() if on else self.eyes.off()

    def rest(self):
        """Close the jaw, then stop sending pulses so the servo does not buzz."""
        self.move(self.closed)
        if self.servo is not None:
            time.sleep(0.15)
            self.servo.angle = None
        self.last_angle = None


# --------------------------------------------------------------------------
# Sample-rate helpers (whole-number ratios only; cheap enough for a Pi 3)
# --------------------------------------------------------------------------
def rate_factor(device_rate):
    if device_rate % SERVER_RATE:
        raise SystemExit("Rate %d is not a multiple of %d. Use 16000, 32000 or 48000."
                         % (device_rate, SERVER_RATE))
    return device_rate // SERVER_RATE


def downsample(samples, factor):
    if factor == 1:
        return samples
    usable = len(samples) - len(samples) % factor
    return samples[:usable].reshape(-1, factor).mean(axis=1).astype(np.int16)


def upsample(samples, factor):
    return samples if factor == 1 else np.repeat(samples, factor)


# --------------------------------------------------------------------------
# The conversation
# --------------------------------------------------------------------------
class TalkingProp:
    def __init__(self, args, jaw):
        self.args = args
        self.jaw = jaw
        self.mic_factor = rate_factor(args.mic_rate)
        self.out_factor = rate_factor(args.out_rate)
        self.mic_queue = queue.Queue(maxsize=50)
        self.play_buffer = bytearray()  # 16 kHz PCM waiting to be played
        self.play_lock = threading.Lock()
        self.last_audio_time = 0.0
        self.is_speaking = False
        self.ws = None
        self.connected = threading.Event()
        self.stop = threading.Event()
        self.response_idle = threading.Event()  # set when no model response is in progress
        self.response_idle.set()
        self.send_lock = threading.Lock()
        self.last_user_text = ""
        self.called_tool = False   # the current response called a tool
        self.promised = False      # the current response said it would look something up

    # ---- audio device callbacks (run on PyAudio's thread) ----
    def on_mic(self, in_data, frame_count, time_info, status):
        try:
            self.mic_queue.put_nowait(in_data)
        except queue.Full:
            pass  # not connected or network stalled: drop audio rather than lag
        return (None, 0)  # 0 == pyaudio.paContinue

    def on_speaker(self, in_data, frame_count, time_info, status):
        want = (frame_count // self.out_factor) * 2  # bytes of 16 kHz audio
        with self.play_lock:
            chunk = bytes(self.play_buffer[:want])
            del self.play_buffer[:want]
        samples = np.frombuffer(chunk, dtype=np.int16)
        if len(samples):
            self.last_audio_time = time.monotonic()
            self.jaw.move(self.jaw.target(samples))
        if len(chunk) < want:
            samples = np.concatenate([samples, np.zeros((want - len(chunk)) // 2, dtype=np.int16)])
        return (upsample(samples, self.out_factor).tobytes(), 0)

    def prop_is_talking(self):
        """True while reply audio is playing, plus a short tail for room echo."""
        with self.play_lock:
            pending = len(self.play_buffer) > 0
        return pending or (time.monotonic() - self.last_audio_time) < self.args.tail

    # ---- websocket ----
    def send(self, message):
        """Send one JSON message; the mic thread and tool threads share the socket."""
        with self.send_lock:
            self.ws.send(json.dumps(message))

    def session_update(self):
        output = {"format": {"type": "audio/pcm", "rate": None}}
        if self.args.voice:
            output["voice"] = self.args.voice
        instructions = self.args.persona_text
        if self.args.kid_safe:
            instructions = KID_SAFE_RULE + "\n\n" + instructions
        now = current_time()
        instructions += ("\n\nRight now it is %s on %s (%s). When asked the time or date, give exactly "
                         "this, in your own words; never guess it." % (now["time"], now["date"], now["timezone"]))
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": instructions,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": None},
                        "transcription": {"model": "gpt-4o-transcribe", "language": self.args.language},
                        "turn_detection": {"type": "server_vad", "interrupt_response": True},
                    },
                    "output": output,
                },
                "tools": [TIME_TOOL] + ([SEARCH_TOOL] if self.args.brave_key else []),
                "tool_choice": "auto",
            },
        }

    def on_open(self, ws):
        self.response_idle.set()
        self.send(self.session_update())
        self.connected.set()
        print("Connected to %s (voice=%s, web search %s)" % (
            self.args.url, self.args.voice or "server default", "on" if self.args.brave_key else "off"))

    def search_results_for(self, query):
        print("Search: %s" % query)
        result = brave_search(self.args.brave_key, query)
        if "error" in result:
            print("Search problem: %s" % result["error"])
        elif self.args.kid_safe:
            result["guidance"] = ("Children are listening. Mention only cheerful, family-friendly items; "
                                  "skip anything about crime, death, violence, war or disasters.")
        return result

    def fallback_search(self, query):
        """The model promised to look something up but didn't call web_search: do it for it."""
        result = self.search_results_for(query)
        self.response_idle.wait(15)
        if not self.connected.is_set():
            return
        note = ("[Web search results for the visitor's question %r, which you said you would look up. "
                "Answer it now from these, briefly and in character. Results: %s]" % (query, json.dumps(result)))
        try:
            self.send({"type": "conversation.item.create",
                       "item": {"type": "message", "role": "user",
                                "content": [{"type": "input_text", "text": note}]}})
            self.send({"type": "response.create"})
        except Exception as exc:
            print("Could not return search results: %s" % exc)

    def run_tool(self, name, arguments, call_id):
        """Run a tool call from the model, then hand the result back and ask it to answer."""
        try:
            params = json.loads(arguments or "{}")
        except ValueError:
            params = {}
        if name == "get_current_time":
            result = current_time()
        elif name == "web_search" and self.args.brave_key and params.get("query"):
            result = self.search_results_for(params["query"])
        else:
            result = {"error": "unknown tool %r" % name}
        # The server takes tool results only after the response that asked for them is done.
        self.response_idle.wait(15)
        if not self.connected.is_set():
            return
        try:
            self.send({"type": "conversation.item.create",
                       "item": {"type": "function_call_output", "call_id": call_id,
                                "output": json.dumps(result)}})
            self.send({"type": "response.create"})
        except Exception as exc:
            print("Could not return search results: %s" % exc)

    def on_message(self, ws, message):
        try:
            event = json.loads(message)
        except ValueError:
            return
        kind = event.get("type", "")
        if kind in ("response.output_audio.delta", "response.audio.delta"):
            pcm = base64.b64decode(event.get("delta", ""))
            with self.play_lock:
                self.play_buffer.extend(pcm)
        elif kind == "input_audio_buffer.speech_started":
            if self.args.barge_in:  # someone spoke over the prop: stop it
                with self.play_lock:
                    del self.play_buffer[:]
        elif kind == "conversation.item.input_audio_transcription.completed":
            self.last_user_text = (event.get("transcript") or "").strip()
            print("You:   %s" % self.last_user_text)
        elif kind in ("response.output_audio_transcript.done", "response.audio_transcript.done"):
            reply = (event.get("transcript") or "").strip()
            print("Prop:  %s" % reply)
            self.promised = promised_to_search(reply)
        elif kind == "response.created":
            self.response_idle.clear()
            self.called_tool = self.promised = False
        elif kind == "response.done":
            self.response_idle.set()
            if (self.promised and not self.called_tool and self.args.brave_key
                    and self.args.search_fallback and self.last_user_text):
                query, self.last_user_text = self.last_user_text, ""
                threading.Thread(target=self.fallback_search, args=(query,), daemon=True).start()
            self.promised = False
        elif kind == "response.function_call_arguments.done":
            self.called_tool = True
            threading.Thread(target=self.run_tool, daemon=True,
                             args=(event.get("name"), event.get("arguments"), event.get("call_id"))).start()
        elif kind == "error":
            print("Server error: %s" % json.dumps(event.get("error", event)))
        elif self.args.debug:
            print("event: %s" % kind)

    def on_close(self, ws, *unused):
        self.connected.clear()

    def on_error(self, ws, error):
        print("Connection problem: %s" % error)

    def run_socket(self):
        import websocket  # the "websocket-client" package

        while not self.stop.is_set():
            self.ws = websocket.WebSocketApp(
                self.args.url,
                header=["Authorization: Bearer %s" % self.args.token],
                on_open=self.on_open,
                on_message=self.on_message,
                on_close=self.on_close,
                on_error=self.on_error,
            )
            self.ws.run_forever(ping_interval=20, ping_timeout=10)
            self.connected.clear()
            if not self.stop.is_set():
                print("Disconnected; retrying in 3 seconds")
                self.stop.wait(3)

    def send_mic(self):
        """Forward mic audio; send silence while the prop talks unless barge-in is on."""
        peak, muted, sent, last_report = 0, False, 0, time.monotonic()
        while not self.stop.is_set():
            try:
                raw = self.mic_queue.get(timeout=0.5)
            except queue.Empty:
                if self.args.debug:
                    print("mic: no audio arriving from the input device")
                continue
            samples = downsample(np.frombuffer(raw, dtype=np.int16), self.mic_factor)
            if self.args.mic_gain != 1.0:
                samples = np.clip(samples.astype(np.float32) * self.args.mic_gain, -32768, 32767).astype(np.int16)
            if self.args.debug:
                if len(samples):
                    peak = max(peak, int(np.abs(samples.astype(np.int32)).max()))
                if time.monotonic() - last_report >= 1.0:
                    print("mic: peak %3d%%  sent=%d chunks%s%s" % (
                        peak * 100 // 32768, sent,
                        "" if self.connected.is_set() else "  (not connected)",
                        "  (muted while talking)" if muted else ""))
                    peak, sent, last_report = 0, 0, time.monotonic()
            if not self.connected.is_set():
                continue
            muted = not self.args.barge_in and self.prop_is_talking()
            if muted:
                samples = np.zeros_like(samples)
            message = {"type": "input_audio_buffer.append",
                       "audio": base64.b64encode(samples.tobytes()).decode("ascii")}
            try:
                self.send(message)
                sent += 1
            except Exception:
                self.connected.clear()

    def refresh_clock(self):
        """Resend the session every so often so the time in the instructions stays current.

        Small models often answer "what time is it?" from their instructions instead of calling
        get_current_time, so the instructions must not go stale. Only sent while idle.
        """
        last = time.monotonic()
        while not self.stop.wait(1.0):
            if time.monotonic() - last < self.args.clock_refresh:
                continue
            if not self.connected.is_set() or not self.response_idle.is_set() or self.prop_is_talking():
                continue
            try:
                self.send(self.session_update())
                last = time.monotonic()
            except Exception:
                pass

    def watch_speaking(self):
        """Light the eyes (if wired) while talking and relax the servo afterwards."""
        while not self.stop.is_set():
            talking = self.prop_is_talking()
            if talking and not self.is_speaking:
                self.jaw.speaking(True)
            elif not talking and self.is_speaking:
                self.jaw.speaking(False)
                self.jaw.rest()
            self.is_speaking = talking
            time.sleep(0.05)

    def run(self):
        import pyaudio

        pa = pyaudio.PyAudio()
        mic_index = find_device(pa, self.args.mic_device, need_input=True)
        out_index = find_device(pa, self.args.out_device, need_input=False)
        mic = pa.open(format=pyaudio.paInt16, channels=1, rate=self.args.mic_rate, input=True,
                      input_device_index=mic_index,
                      frames_per_buffer=int(self.args.mic_rate * 0.04), stream_callback=self.on_mic)
        speaker = pa.open(format=pyaudio.paInt16, channels=1, rate=self.args.out_rate, output=True,
                          output_device_index=out_index,
                          frames_per_buffer=int(self.args.out_rate * 0.02), stream_callback=self.on_speaker)
        threads = [threading.Thread(target=fn, daemon=True)
                   for fn in (self.run_socket, self.send_mic, self.watch_speaking, self.refresh_clock)]
        for thread in threads:
            thread.start()
        print("Listening. Ctrl-C to quit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set()
            if self.ws is not None:
                self.ws.close()
            mic.stop_stream()
            speaker.stop_stream()
            mic.close()
            speaker.close()
            pa.terminate()
            self.jaw.rest()


def find_device(pa, wanted, need_input):
    """Turn a device number or a piece of its name into a PyAudio device index."""
    if wanted is None:
        return None
    key = "maxInputChannels" if need_input else "maxOutputChannels"
    devices = [pa.get_device_info_by_index(i) for i in range(pa.get_device_count())]
    if str(wanted).isdigit():
        index = int(wanted)
        if index < len(devices) and devices[index][key] > 0:
            return index
        raise SystemExit("Device %s is not a usable %s. Run --list-devices; numbers can change after a reboot."
                         % (wanted, "microphone" if need_input else "speaker"))
    usable = [d for d in devices if d[key] > 0]
    exact = [d for d in usable if d["name"].lower() == str(wanted).lower()]
    partial = [d for d in usable if str(wanted).lower() in d["name"].lower()]
    for match in exact + partial:
        return int(match["index"])
    raise SystemExit("No %s matching %r. Run --list-devices." % ("microphone" if need_input else "speaker", wanted))


def list_devices():
    import pyaudio

    pa = pyaudio.PyAudio()
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        print("%2d  in=%d out=%d  %s" % (index, info["maxInputChannels"],
                                          info["maxOutputChannels"], info["name"]))
    pa.terminate()


def jaw_test(jaw):
    """Step the jaw through its four positions so the calibration can be checked."""
    step = (jaw.open - jaw.closed) / 3.0
    for _ in range(3):
        for position in (0, 1, 2, 3, 2, 1):
            jaw.move(jaw.closed + position * step)
            time.sleep(0.25)
    jaw.rest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Live voice conversation for an animatronic prop.")
    p.add_argument("--url", default=os.getenv("TALKING_PROP_URL", ""),
                   help="realtime websocket URL of the s2s server, e.g. ws://192.168.1.50:8765/v1/realtime "
                        "(or set TALKING_PROP_URL)")
    p.add_argument("--token", default=os.getenv("TALKING_PROP_TOKEN", "DUMMY"),
                   help="bearer token, if the server wants one")
    p.add_argument("--voice", default=os.getenv("TALKING_PROP_VOICE", ""), help="voice name known to the s2s server")
    p.add_argument("--language", default="en", help="transcription language")
    p.add_argument("--persona", default=os.path.join(HERE, "persona.txt"), help="file with the personality text")
    p.add_argument("--config", default=os.path.join(HERE, "config.ini"), help="ChatterPi config.ini to reuse")
    p.add_argument("--mic-rate", type=int, default=16000, help="mic sample rate: 16000, 32000 or 48000")
    p.add_argument("--out-rate", type=int, default=16000, help="speaker sample rate: 16000, 32000 or 48000")
    p.add_argument("--mic-device", default=None, help="input device: a number, or part of its name (see --list-devices)")
    p.add_argument("--out-device", default=None, help="output device: a number, or part of its name (see --list-devices)")
    p.add_argument("--barge-in", action="store_true",
                   help="keep the mic live while the prop talks (needs an echo-cancelling mic)")
    p.add_argument("--mic-gain", type=float, default=1.0,
                   help="multiply the mic signal, e.g. 3 for a quiet mic (try the mixer level first)")
    p.add_argument("--tail", type=float, default=0.4, help="seconds to keep the mic muted after the prop stops")
    p.add_argument("--no-servo", action="store_true", help="run without touching GPIO")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--jaw-test", action="store_true", help="step the jaw through its positions and exit")
    p.add_argument("--brave-key-file", default=os.path.join(HERE, "brave_key.txt"),
                   help="file holding the Brave Search API key (or set BRAVE_API_KEY)")
    p.add_argument("--no-search", action="store_true", help="turn web search off even if a key is present")
    p.add_argument("--no-search-fallback", dest="search_fallback", action="store_false",
                   help="don't search on the model's behalf when it says it will look something up but doesn't")
    p.add_argument("--kid-safe", action="store_true",
                   help="keep answers and search results family-friendly (for props around children)")
    p.add_argument("--clock-refresh", type=float, default=60.0,
                   help="seconds between updates of the time given to the model (default 60)")
    p.add_argument("--debug", action="store_true", help="print server events and a mic level meter once a second")
    args = p.parse_args(argv)
    args.brave_key = "" if args.no_search else load_brave_key(args.brave_key_file)
    args.persona_text = DEFAULT_PERSONA
    if os.path.isfile(args.persona):
        with open(args.persona, encoding="utf-8") as handle:
            args.persona_text = handle.read().strip() or DEFAULT_PERSONA
    return args


def main():
    args = parse_args()
    if args.list_devices:
        return list_devices()
    jaw = Jaw(load_chatterpi_config(args.config), enabled=not args.no_servo)
    if args.jaw_test:
        return jaw_test(jaw)
    if not args.url:
        raise SystemExit("No server URL. Pass --url ws://<server>:8765/v1/realtime or set TALKING_PROP_URL.")
    TalkingProp(args, jaw).run()


if __name__ == "__main__":
    main()
