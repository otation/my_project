import cv2
import numpy as np
import pyaudio
import librosa
import time
import logging
import sys
from datetime import datetime
import mediapipe as mp
from collections import deque
from enum import Enum

# 음계 정의
class Note(Enum):
    C4 = ("C4", "도", 60)   # (음표, 한글이름, MIDI number)
    D4 = ("D4", "레", 62)
    E4 = ("E4", "미", 64)
    F4 = ("F4", "파", 65)
    G4 = ("G4", "솔", 67)
    A4 = ("A4", "라", 69)
    B4 = ("B4", "시", 71)

# 피아노 건반 클래스
class PianoKey:
    def __init__(self, note, x, width, is_black=False):
        self.note = note
        self.x = x
        self.width = width
        self.is_black = is_black
        self.is_active = False

class Piano:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.white_key_width = width // 7
        self.black_key_width = self.white_key_width * 0.6
        self.black_key_height = height * 0.6
        self.keys = []
        self.init_keys()
        
    def init_keys(self):
        # 흰 건반 초기화
        white_notes = [Note.C4, Note.D4, Note.E4, Note.F4, Note.G4, Note.A4, Note.B4]
        for i, note in enumerate(white_notes):
            x = self.x + i * self.white_key_width
            self.keys.append(PianoKey(note, x, self.white_key_width))

    def draw(self, frame, current_note):
        # 흰 건반 그리기
        for key in [k for k in self.keys if not k.is_black]:
            x1 = int(key.x)
            x2 = int(key.x + key.width)
            y1 = self.y
            y2 = self.y + self.height
            
            # 현재 음표와 일치하는 건반 강조
            color = (0, 255, 0) if key.note == current_note else (255, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
            
            # 음표 이름 표시
            if key.note:
                cv2.putText(frame, key.note.value[1], 
                           (x1 + 5, y2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

class InputModule:
    def __init__(self):
        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("웹캠을 열 수 없습니다.")
        
        self.frame_metrics = {
            'timestamp': None,
            'frame_size': None,
            'frame_rate': 0
        }
        
        # 오디오 초기화
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=44100,
            input=True,
            frames_per_buffer=1024
        )
        self.audio_metrics = {
            'timestamp': None,
            'chunk_size': 1024,
            'sample_rate': 44100
        }

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_metrics['timestamp'] = time.time()
            self.frame_metrics['frame_size'] = frame.shape
            return frame, self.frame_metrics
        return None, None

    def capture_audio(self):
        try:
            audio_data = np.frombuffer(
                self.stream.read(1024, exception_on_overflow=False),
                dtype=np.float32
            )
            self.audio_metrics['timestamp'] = time.time()
            return audio_data, self.audio_metrics
        except Exception as e:
            logging.error(f"오디오 캡처 실패: {str(e)}")
            return np.zeros(1024, dtype=np.float32), self.audio_metrics

    def __del__(self):
        self.cap.release()
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()

class ProcessingModule:
    def __init__(self):
        # MediaPipe 초기화
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 학교종이 땡땡땡 악보
        self.song_notes = deque([
            Note.G4, Note.G4, Note.A4, Note.A4,
            Note.G4, Note.G4, Note.E4,
            Note.G4, Note.G4, Note.E4, Note.E4,
            Note.D4, Note.D4, Note.C4
        ])
        self.current_note = self.song_notes[0]
        
        # 음계 매칭을 위한 주파수 범위
        self.note_frequencies = {
            Note.C4: (261.63, 15),  # (기준 주파수, 허용 오차 Hz)
            Note.D4: (293.66, 15),
            Note.E4: (329.63, 15),
            Note.F4: (349.23, 15),
            Note.G4: (392.00, 15),
            Note.A4: (440.00, 15),
            Note.B4: (493.88, 15)
        }
        
        # 매칭 상태 관리
        self.match_start_time = None
        self.required_match_duration = 0.3
        self.last_matched_note = None
        self.note_changed_time = time.time()

    def preprocess_image(self, frame):
        try:
            if frame is None:
                raise ValueError("입력 프레임이 None입니다")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error(f"이미지 전처리 실패: {str(e)}")
            return None

    def detect_hand(self, frame):
        try:
            if frame is None:
                return None
            results = self.hands.process(frame)
            return results
        except Exception as e:
            logging.error(f"손 감지 실패: {str(e)}")
            return None

    def track_position(self, detection_results, frame_shape):
        try:
            if not detection_results or not detection_results.multi_hand_landmarks:
                return []
            
            detections = []
            height, width = frame_shape[:2]
            
            for hand_landmarks in detection_results.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                x = int(index_tip.x * width)
                y = int(index_tip.y * height)
                
                detections.append({
                    'center': (x, y),
                    'confidence': 0.9,
                    'landmarks': hand_landmarks
                })
            return detections
        except Exception as e:
            logging.error(f"위치 추적 실패: {str(e)}")
            return []

    def preprocess_audio(self, audio_data):
        try:
            if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
                return np.zeros_like(audio_data)
            return librosa.util.normalize(audio_data)
        except Exception as e:
            logging.error(f"오디오 전처리 실패: {str(e)}")
            return np.zeros_like(audio_data)

    def detect_pitch(self, audio_data):
        try:
            if np.all(audio_data == 0):
                return np.array([]), np.array([])

            pitches, magnitudes = librosa.piptrack(
                y=audio_data,
                sr=44100,
                n_fft=2048,
                hop_length=512,
                fmin=50,
                fmax=2000
            )
            return pitches, magnitudes
        except Exception as e:
            logging.error(f"피치 검출 실패: {str(e)}")
            return np.array([]), np.array([])

    def is_note_match(self, detected_frequency, target_note):
        if target_note not in self.note_frequencies:
            return False
        target_freq, tolerance = self.note_frequencies[target_note]
        return abs(detected_frequency - target_freq) <= tolerance

    def get_frequency_from_pitch(self, pitch):
        if pitch <= 0:
            return None
        return pitch

    def recognize_note(self, pitches, magnitudes):
        try:
            if pitches.size == 0 or magnitudes.size == 0:
                self.match_start_time = None
                self.last_matched_note = None
                return None, None  # 주파수도 함께 반환

            max_magnitude_indices = magnitudes.argmax(axis=0)
            pitches_with_max_magnitude = pitches[max_magnitude_indices, range(pitches.shape[1])]
            current_pitch = pitches_with_max_magnitude[0]
            
            detected_freq = self.get_frequency_from_pitch(current_pitch)
            if detected_freq is None:
                self.match_start_time = None
                return None, None

            is_matching = self.is_note_match(detected_freq, self.current_note)
            current_time = time.time()

            if is_matching:
                if self.match_start_time is None:
                    self.match_start_time = current_time
                    self.last_matched_note = self.current_note
                elif (current_time - self.match_start_time >= self.required_match_duration and 
                      self.last_matched_note == self.current_note):
                    logging.info(f"음계 매칭 성공: {self.current_note.value[1]}")
                    self.song_notes.rotate(-1)
                    self.current_note = self.song_notes[0]
                    self.match_start_time = None
                    self.last_matched_note = None
                    self.note_changed_time = current_time
            else:
                self.match_start_time = None
                self.last_matched_note = None

            detected_note = librosa.hz_to_note(detected_freq)
            return detected_note, detected_freq  # 주파수도 함께 반환

        except Exception as e:
            logging.error(f"음계 인식 실패: {str(e)}")
            return None, None

    def get_match_progress(self):
        if self.match_start_time is None:
            return 0.0
        progress = (time.time() - self.match_start_time) / self.required_match_duration
        return min(progress, 1.0)

    def __del__(self):
        self.hands.close()

class IntegrationModule:
    def __init__(self, processing_module):
        self.mp_drawing = mp.solutions.drawing_utils  # drawing_utils로 변경
        self.mp_hands = mp.solutions.hands
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2)  # mp_drawing 사용
        self.piano = None
        self.processing_module = processing_module

    def create_overlay(self, frame, hand_detections, current_note, detected_note, current_freq=None):
        try:
            overlay = frame.copy()
            height, width = overlay.shape[:2]
            
            if self.piano is None:
                piano_width = width - 40
                piano_height = height // 4
                self.piano = Piano(20, height - piano_height - 20, piano_width, piano_height)
            
            self.piano.draw(overlay, current_note)
            
            for detection in hand_detections:
                if 'landmarks' in detection:
                    self.mp_drawing.draw_landmarks(
                        overlay,
                        detection['landmarks'],
                        self.mp_hands.HAND_CONNECTIONS,
                        self.drawing_spec,
                        self.drawing_spec
                    )
                    
                    x, y = detection['center']
                    cv2.circle(overlay, (x, y), 8, (0, 0, 255), -1)
            
            progress = self.processing_module.get_match_progress()
            if progress > 0:
                bar_width = 200
                bar_height = 20
                filled_width = int(bar_width * progress)
                
                cv2.rectangle(overlay, 
                            (width - bar_width - 20, 20), 
                            (width - 20, 20 + bar_height),
                            (100, 100, 100), 
                            1)
                
                cv2.rectangle(overlay,
                            (width - bar_width - 20, 20),
                            (width - bar_width - 20 + filled_width, 20 + bar_height),
                            (0, 255, 0),
                            -1)
            
            # 현재 음표와 주파수 정보 표시
            cv2.putText(overlay, 
                       f"Current Note: {current_note.value[1]}", 
                       (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2)
            
            if detected_note:
                cv2.putText(overlay, 
                           f"Detected: {detected_note}", 
                           (20, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, (255, 0, 0), 2)
            
            # 현재 주파수 표시
            if current_freq is not None:
                cv2.putText(overlay,
                           f"Frequency: {current_freq:.1f} Hz",
                           (20, 90),  # 음표 아래에 표시
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1.0,
                           (255, 165, 0),  # 주황색
                           2)
                
                # 목표 주파수와 허용 범위 표시
                target_freq, tolerance = self.processing_module.note_frequencies[current_note]
                cv2.putText(overlay,
                           f"Target: {target_freq:.1f} Hz (±{tolerance} Hz)",
                           (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           1.0,
                           (0, 255, 255),  # 노란색
                           2)
            
            return overlay
        except Exception as e:
            logging.error(f"오버레이 생성 실패: {str(e)}")
            return frame

    def combine_and_display(self, frame, overlay):
        try:
            result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
            cv2.imshow('Music AR System', result)
            return result
        except Exception as e:
            logging.error(f"화면 표시 실패: {str(e)}")
            cv2.imshow('Music AR System', frame)
            return frame

class ValidationModule:
    def __init__(self):
        self.metrics = {
            'fps': [],
            'latency': [],
            'detection_confidence': [],
            'note_accuracy': []
        }
        self.start_time = time.time()

    def update_metrics(self, start_time, end_time, detection_conf, note_conf):
        latency = end_time - start_time
        self.metrics['latency'].append(latency)
        self.metrics['detection_confidence'].append(detection_conf)
        self.metrics['note_accuracy'].append(note_conf)
        
        fps = 1.0 / latency if latency > 0 else 0
        self.metrics['fps'].append(fps)

    def check_accuracy(self):
        return {
            'average_latency': np.mean(self.metrics['latency']),
            'average_fps': np.mean(self.metrics['fps']),
            'average_detection_confidence': np.mean(self.metrics['detection_confidence']),
            'average_note_accuracy': np.mean(self.metrics['note_accuracy'])
        }

class MusicARSystem:
    def __init__(self):
        self.input_module = InputModule()
        self.processing_module = ProcessingModule()
        self.integration_module = IntegrationModule(self.processing_module)
        self.validation_module = ValidationModule()
        logging.info("시스템 초기화 완료")

    def run(self):
        try:
            while True:
                start_time = time.time()

                frame, frame_metrics = self.input_module.capture_frame()
                if frame is None:
                    continue

                processed_frame = self.processing_module.preprocess_image(frame)
                if processed_frame is None:
                    continue
                    
                hand_results = self.processing_module.detect_hand(processed_frame)
                hand_detections = self.processing_module.track_position(hand_results, frame.shape)

                try:
                    audio_data, _ = self.input_module.capture_audio()
                    processed_audio = self.processing_module.preprocess_audio(audio_data)
                    pitches, magnitudes = self.processing_module.detect_pitch(processed_audio)
                    detected_note, current_freq = self.processing_module.recognize_note(pitches, magnitudes)  # 주파수도 받아옴
                except Exception as e:
                    logging.error(f"오디오 처리 오류: {str(e)}")
                    detected_note, current_freq = None, None

                overlay = self.integration_module.create_overlay(
                    frame, 
                    hand_detections, 
                    self.processing_module.current_note,
                    detected_note,
                    current_freq  # 주파수 전달
                )
                self.integration_module.combine_and_display(frame, overlay)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            logging.error(f"실행 중 오류 발생: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()
        logging.info("시스템 종료")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'music_ar_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("프로그램 시작")

    try:
        system = MusicARSystem()
        system.run()
    except Exception as e:
        logger.error(f"시스템 초기화 실패: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        logger.info("프로그램 종료")

if __name__ == "__main__":
    main()