#!/usr/bin/env python3
"""Live brick dashboard using the existing remote SAM3 and Approach 17 model."""
import argparse
import csv
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk

import cv2
from remote_rgb_client import (CSV_FIELDS, LAB_PYTHON, LAB_REPO, RemoteModel,
                               annotate_response, open_camera, save_result, timestamp)

BG, PANEL, TEXT, MUTED = '#101720', '#1a2531', '#f2f6f7', '#a9bac7'
GREEN, RED, AMBER, BLUE = '#55d6a0', '#ff7770', '#f4c267', '#74baf4'


class Dashboard:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.model = self.camera = self.latest = None
        self.photos = {}
        self.jobs, self.messages = queue.Queue(maxsize=1), queue.Queue()
        self.stop = threading.Event()
        self.processing = self.fullscreen = self.closing = False
        self.frame_id, self.next_capture = 0, 0
        self.missing_since = None
        self.camera_waiting = False
        self.connection_started = time.monotonic()
        self.completed_frames = 0
        self.output = args.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.output.mkdir(parents=True, exist_ok=False)
        self.stream = (self.output / 'predictions.csv').open('w', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=CSV_FIELDS)
        self.writer.writeheader()
        self.stream.flush()
        self.build()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.bind('<Escape>', self.escape)
        root.bind('<KeyPress-f>', lambda e: self.toggle_fullscreen())
        root.bind('<KeyPress-F>', lambda e: self.toggle_fullscreen())
        try:
            self.camera = open_camera(args)
        except Exception as exc:
            self.camera_status.config(text='●  Camera error', fg=RED)
            self.details.config(text=str(exc), fg=RED)
            return
        threading.Thread(target=self.inference_loop, daemon=True).start()
        root.after(20, self.tick)

    def build(self):
        self.root.title('Brick inspection')
        self.root.configure(bg=BG)
        self.root.geometry('1440x850')
        self.root.minsize(850, 570)
        head = tk.Frame(self.root, bg=BG, padx=24, pady=18)
        head.pack(fill='x')
        title = tk.Frame(head, bg=BG)
        title.pack(side='left')
        tk.Label(title, text='BRICK INSPECTION', fg=TEXT, bg=BG,
                 font=('Helvetica', 22, 'bold')).pack(anchor='w')
        tk.Label(title, text='Live camera  /  SAM3  /  Approach 17', fg=MUTED, bg=BG,
                 font=('Helvetica', 11)).pack(anchor='w', pady=(4, 0))
        tk.Button(head, text='Fullscreen  [ F ]', command=self.toggle_fullscreen).pack(side='right')
        self.status = tk.Label(head, text='●  Connecting to model…', fg=AMBER, bg=BG,
                               font=('Helvetica', 11, 'bold'))
        self.status.pack(side='right', padx=15)
        self.camera_status = tk.Label(head, text='●  Camera starting', fg=AMBER, bg=BG,
                                      font=('Helvetica', 11, 'bold'))
        self.camera_status.pack(side='right', padx=15)

        cards = tk.Frame(self.root, bg=BG, padx=24, pady=4)
        cards.pack(fill='x')
        self.metrics = {}
        for label, color in (('BRICKS IN FRAME', BLUE), ('GOOD', GREEN), ('BAD', RED), ('RESPONSE', TEXT)):
            card = tk.Frame(cards, bg=PANEL, padx=16, pady=8)
            card.pack(side='left', fill='x', expand=True, padx=5)
            tk.Label(card, text=label, bg=PANEL, fg=MUTED, font=('Helvetica', 9, 'bold')).pack(anchor='w')
            value = tk.Label(card, text='—', bg=PANEL, fg=color, font=('Helvetica', 20, 'bold'))
            value.pack(anchor='w')
            self.metrics[label] = value
        content = tk.Frame(self.root, bg=BG, padx=24)
        content.pack(fill='both', expand=True)
        content.columnconfigure(0, weight=1, uniform='video')
        content.columnconfigure(1, weight=1, uniform='video')
        content.rowconfigure(1, weight=1)
        for col, (title, caption) in enumerate((('LIVE CAMERA', 'Current camera feed'),
                                                 ('PREDICTIONS', 'Most recently analyzed frame'))):
            bar = tk.Frame(content, bg=BG)
            bar.grid(row=0, column=col, sticky='ew', padx=7, pady=(0, 8))
            tk.Label(bar, text=title, fg=TEXT, bg=BG, font=('Helvetica', 12, 'bold')).pack(side='left')
            tk.Label(bar, text=caption, fg=MUTED, bg=BG, font=('Helvetica', 10)).pack(side='right')
        self.live = tk.Label(content, text='Opening camera…', fg=MUTED, bg=PANEL)
        self.live.grid(row=1, column=0, sticky='nsew', padx=7)
        self.prediction = tk.Label(content, text='Waiting for first prediction…', fg=MUTED, bg=PANEL)
        self.prediction.grid(row=1, column=1, sticky='nsew', padx=7)
        bottom = tk.Frame(self.root, bg=BG, padx=24, pady=16)
        bottom.pack(fill='x')
        tk.Label(bottom, text='LATEST INSPECTION', bg=BG, fg=TEXT,
                 font=('Helvetica', 12, 'bold')).pack(anchor='w')
        self.table = tk.Frame(bottom, bg=PANEL, padx=10, pady=5)
        self.table.pack(fill='x', pady=(6, 0))
        self.rows = []
        for i in range(2):
            row = tk.Label(self.table, text='—', bg=PANEL, fg=MUTED, anchor='w',
                           font=('Courier', 10), justify='left')
            row.pack(fill='x', pady=1)
            self.rows.append(row)
        self.details = tk.Label(bottom, text='No completed inspection yet', bg=BG, fg=MUTED,
                                font=('Helvetica', 11), justify='left', wraplength=1300)
        self.details.pack(anchor='w', pady=(8, 0))
        tk.Label(bottom, text=f'Policy: good ≤ −2.0  ·  bad > −2.0    |    Every {self.args.interval:g}s    |    Results: {self.output}',
                 fg=MUTED, bg=BG, font=('Helvetica', 10), wraplength=1300, justify='left').pack(anchor='w', pady=(10, 0))

    def display_result(self, record):
        self.completed_frames += 1
        for name, key in (('BRICKS IN FRAME', 'brick_count'), ('GOOD', 'good_count'), ('BAD', 'bad_count')):
            self.metrics[name].config(text=str(record.get(key, 0)))
        elapsed = record.get('roundtrip_seconds')
        self.metrics['RESPONSE'].config(text=f'{elapsed:.1f} s' if isinstance(elapsed, (int, float)) else '—')
        detections = record.get('detections', [])
        parts = []
        for d in detections:
            score = d.get('score_bad')
            if isinstance(score, (int, float)):
                parts.append(f"#{d.get('id', '?')} {str(d.get('label', '?')).upper()}  {score:+.2f}")
        for i, row in enumerate(self.rows):
            values = parts[i * 5:(i + 1) * 5]
            row.config(text='    |    '.join(values) if values else ('No reliable bricks detected' if not parts and i == 0 else ' '))
        captured = record.get('captured_at', '—')
        inference = record.get('inference_seconds')
        inf_text = f'{inference:.1f}s' if isinstance(inference, (int, float)) else '—'
        branches = ''
        if len(detections) == 1:
            a5, dino = detections[0].get('approach5_score'), detections[0].get('dinov2_score')
            if isinstance(a5, (int, float)) and isinstance(dino, (int, float)):
                branches = f'  ·  A5 {a5:+.2f}  ·  DINOv2 {dino:+.2f}'
        self.details.config(text=f'Frame {record.get("frame_id", "—")}  ·  captured {captured}  ·  model {inf_text}  ·  session frames {self.completed_frames}{branches}   |   Scores are decision values, not probabilities.', fg=MUTED)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes('-fullscreen', self.fullscreen)

    def escape(self, event=None):
        if self.fullscreen:
            self.toggle_fullscreen()

    def inference_loop(self):
        model = None
        try:
            model = RemoteModel(self.args)
            self.model = model
            if self.stop.is_set():
                return
            self.messages.put(('ready', None))
            while not self.stop.is_set():
                try:
                    frame, frame_id, captured = self.jobs.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    record = model.predict(frame, frame_id, captured, None, self.args.jpeg_quality)
                    self.messages.put(('result', {'raw': frame, 'annotated': annotate_response(frame, record), 'record': record}))
                finally:
                    self.jobs.task_done()
        except Exception as exc:
            if not self.stop.is_set():
                self.messages.put(('error', f'{type(exc).__name__}: {exc}'))
        finally:
            if model is not None:
                model.close()

    def show_image(self, widget, image, name):
        h, w = image.shape[:2]
        scale = min(max(100, widget.winfo_width()) / w, max(100, widget.winfo_height()) / h)
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        photo = tk.PhotoImage(data=f'P6\n{w} {h}\n255\n'.encode() + rgb.tobytes(), format='PPM')
        self.photos[name] = photo
        widget.configure(image=photo, text='')

    def tick(self):
        if self.closing:
            return
        try:
            while True:
                try:
                    kind, value = self.messages.get_nowait()
                except queue.Empty:
                    break
                if kind == 'ready':
                    self.status.config(text='●  Connected · inspecting', fg=GREEN)
                elif kind == 'error':
                    self.processing = False
                    self.status.config(text='●  Connection or inference error', fg=RED)
                    self.details.config(text=value, fg=RED)
                    self.stop.set()
                else:
                    self.processing = False
                    self.latest = value
                    save_result(value, self.writer, self.stream, self.output, self.args.save_frames)
                    record = value['record']
                    self.status.config(text='●  Model connected', fg=GREEN)
                    self.display_result(record)
            ok, frame = self.camera.read()
            if ok:
                self.camera_status.config(text='●  Camera live', fg=GREEN)
                self.missing_since = None
                if self.camera_waiting:
                    self.camera_waiting = False
                    if not self.stop.is_set():
                        self.details.config(text='Camera resumed. Waiting for the next inspection.', fg=MUTED)
                self.show_image(self.live, frame, 'live')
                if self.latest is not None:
                    self.show_image(self.prediction, self.latest['annotated'], 'prediction')
                now = time.monotonic()
                if self.model is not None and not self.stop.is_set() and not self.processing and now >= self.next_capture:
                    self.frame_id += 1
                    self.jobs.put_nowait((frame.copy(), self.frame_id, timestamp()))
                    self.processing = True
                    self.status.config(text='●  Model processing frame', fg=BLUE)
                    self.next_capture = now + self.args.interval
            else:
                now = time.monotonic()
                if self.missing_since is None:
                    self.missing_since = now
                if now - self.missing_since >= 5 and not self.camera_waiting:
                    self.camera_waiting = True
                    if not self.stop.is_set():
                        self.camera_status.config(text='●  Waiting for camera frames', fg=AMBER)
                        self.details.config(text='Camera is open but no new frames arrived. Retrying automatically; check the camera connection if this continues.', fg=AMBER)
            if self.model is None and not self.stop.is_set():
                elapsed = int(time.monotonic() - self.connection_started)
                self.status.config(text=f'●  Model connecting · {elapsed}s', fg=AMBER)
        except Exception as exc:
            self.status.config(text='●  Viewer error', fg=RED)
            self.details.config(text=f'{type(exc).__name__}: {exc}', fg=RED)
            self.stop.set()
            return
        self.root.after(50, self.tick)

    def close(self):
        self.closing = True
        self.stop.set()
        if self.camera is not None:
            self.camera.release()
        self.stream.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='131.220.238.95')
    parser.add_argument('--user', default='s95vkond')
    parser.add_argument('--remote-repo', default=LAB_REPO)
    parser.add_argument('--remote-python', default=LAB_PYTHON)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--backend', choices=('auto', 'avfoundation', 'dshow', 'v4l2'), default='avfoundation')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--interval', type=float, default=3.0)
    parser.add_argument('--jpeg-quality', type=int, default=92)
    parser.add_argument('--output', type=Path, default=Path.cwd() / 'gui_results')
    parser.add_argument('--save-frames', action='store_true')
    parser.add_argument('--fullscreen', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0 or args.width < 1 or args.height < 1 or not 60 <= args.jpeg_quality <= 100:
        parser.error('Check interval, camera dimensions and JPEG quality (60–100)')
    args.pipeline, args.policy = 'sam3', 'minus_two_binary'
    root = tk.Tk()
    app = Dashboard(root, args)
    if args.fullscreen:
        app.toggle_fullscreen()
    root.mainloop()


if __name__ == '__main__':
    main()
