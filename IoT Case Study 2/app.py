import tensorflow as tf
import tensorflow_hub as hub
import librosa
import soundfile as sf
import tempfile
import io


from flask import Flask, request, jsonify, render_template
import cv2
import base64
from flask_socketio import SocketIO, emit
import json
import logging
import threading
import time
from datetime import datetime
import base64
import numpy as np
import cv2

# Import our custom modules
from models import ModelManager
from processors import AudioProcessor, VisionProcessor
from responses import ResponseGenerator
import utils

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart_monitoring_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
config = None
model_manager = None
audio_processor = None
vision_processor = None
response_generator = None
logger = None

def initialize_system():
    """Initialize all system components"""
    global config, model_manager, audio_processor, vision_processor, response_generator, logger
    
    # Setup logging
    utils.setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = utils.load_config()
    if not config:
        logger.error("Failed to load configuration")
        return False
    
    # Skip dummy models - using real pre-trained models now
    logger.info("Using real pre-trained models (YAMNet, etc.)")
    model_manager = None  # Not needed for pre-trained models
    
    # Initialize response generator (no processors needed for pre-trained models)
    response_generator = ResponseGenerator(config)
    audio_processor = None  # Using YAMNet directly
    vision_processor = None  # Using pre-trained models directly
    
    logger.info("Pre-trained model system initialized successfully")
    return True

# API Routes
@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Smart Monitoring System API',
        'status': 'running',
        'mobile_interface': '/mobile',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/status')
def status():
    """System status endpoint"""
    system_status = utils.check_system_status()
    system_status['models_loaded'] = model_manager.is_models_loaded() if model_manager else False
    system_status['timestamp'] = datetime.now().isoformat()
    
    return jsonify(system_status)

@app.route('/mobile')
def mobile_interface():
    """Mobile web interface for real sensor access"""
    return render_template('mobile.html')

@app.route('/test-audio', methods=['POST'])
def test_audio():
    """Test audio processing endpoint"""
    try:
        # Generate dummy audio for testing
        dummy_audio = np.random.rand(16000) * 0.6
        
        # Process audio
        result = audio_processor.process_audio_chunk(dummy_audio)
        
        if result:
            # Generate response
            announcement = response_generator.create_announcement_data(
                result['event_type'], 
                result['confidence']
            )
            
            return jsonify({
                'detection': result,
                'announcement': announcement,
                'status': 'success'
            })
        else:
            return jsonify({'error': 'Audio processing failed'}), 500
            
    except Exception as e:
        logger.error(f"Test audio error: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/test-vision', methods=['POST'])
def test_vision():
    """Test vision processing endpoint"""
    try:
        # Generate dummy image for testing
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Process image
        result = vision_processor.process_frame(dummy_image)
        
        if result:
            # Generate response
            announcement = response_generator.create_announcement_data(
                result['event_type'], 
                result['confidence']
            )
            
            return jsonify({
                'detection': result,
                'announcement': announcement,
                'status': 'success'
            })
        else:
            return jsonify({'error': 'Vision processing failed'}), 500
            
    except Exception as e:
        logger.error(f"Test vision error: {e}")
        return jsonify({'error': str(e)}), 500


# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Mobile client connected')
    emit('response', {'message': 'Connected to Smart Monitoring System'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Mobile client disconnected')

@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming real audio data with YAMNet pre-trained model"""
    try:
        audio_array = data.get('data', [])
        timestamp = data.get('timestamp')
        sample_rate = data.get('sample_rate', 16000)
        
        logger.info(f"Processing with YAMNet: {len(audio_array)} samples")
        
        audio_data = np.array(audio_array, dtype=np.float32)
        
        if len(audio_data) > 0:
            event_type, confidence, yamnet_class = classify_audio_with_yamnet(audio_data, sample_rate)
            
            logger.info(f"YAMNet: {yamnet_class} -> {event_type} (conf: {confidence:.3f})")
            
            thresholds = config.get('thresholds', {})
            threshold = thresholds.get(event_type, 0.5)
            
            if confidence >= threshold and event_type != 'background_noise':
                announcement = response_generator.create_announcement_data(
                    event_type, confidence
                )
                
                if announcement:
                    emit('announcement', announcement)
                    logger.info(f"YAMNET DETECTION: {announcement['message']}")
        
    except Exception as e:
        logger.error(f"YAMNet error: {e}")
        emit('error', {'message': 'YAMNet processing failed'})




@socketio.on('image_data')
def handle_image_data(data):
    """Handle incoming real image data with MobileNet pre-trained model"""
    try:
        image_data_base64 = data.get('data', '')
        timestamp = data.get('timestamp')
        
        logger.info(f"Processing with MobileNet and face detection models")
        
        if image_data_base64.startswith('data:image'):
            image_data_base64 = image_data_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_data_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is not None:
            event_type, confidence, detection_details = classify_image_with_pretrained_models(image)
            
            logger.info(f"Vision Models: {detection_details} -> {event_type} (conf: {confidence:.3f})")
            
            thresholds = config.get('thresholds', {})
            threshold = thresholds.get(event_type, 0.5)
            
            if confidence >= threshold and event_type != 'normal':
                announcement = response_generator.create_announcement_data(
                    event_type, confidence
                )
                
                if announcement:
                    emit('announcement', announcement)
                    logger.info(f"VISION MODEL DETECTION: {announcement['message']}")
        
    except Exception as e:
        logger.error(f"Vision model processing error: {e}")
        emit('error', {'message': 'Vision model processing failed'})



@socketio.on('ping')
def handle_ping():
    """Handle ping from mobile"""
    emit('pong', {'timestamp': datetime.now().isoformat()})

        
@socketio.on('simulate_audio')
def handle_simulate_audio(data):
    """Handle simulated audio events"""
    try:
        event_type = data.get('event_type', 'fire_alarm')
        timestamp = data.get('timestamp')
        
        logger.info(f"Simulating audio event: {event_type}")
        
        # Create simulated detection result
        result = {
            'type': 'audio_detection',
            'event_type': event_type,
            'confidence': 0.85,  # High confidence for demo
            'timestamp': timestamp,
            'processor': 'audio_simulation'
        }
        
        # Generate announcement
        announcement = response_generator.create_announcement_data(
            result['event_type'], 
            result['confidence']
        )
        
        if announcement:
            emit('announcement', announcement)
            logger.info(f"Sent simulated announcement: {announcement['message']}")
        
    except Exception as e:
        logger.error(f"Simulate audio error: {e}")
        emit('error', {'message': 'Audio simulation failed'})

@socketio.on('simulate_vision')
def handle_simulate_vision(data):
    """Handle simulated vision events"""
    try:
        event_type = data.get('event_type', 'unknown_person')
        timestamp = data.get('timestamp')
        
        logger.info(f"Simulating vision event: {event_type}")
        
        # Create simulated detection result
        result = {
            'type': 'vision_detection',
            'event_type': event_type,
            
            'confidence': 0.80,  # High confidence for demo
            'timestamp': timestamp,
            'processor': 'vision_simulation'
        }
        
        # Generate announcement
        announcement = response_generator.create_announcement_data(
            result['event_type'], 
            result['confidence']
        )
        
        if announcement:
            emit('announcement', announcement)
            logger.info(f"Sent simulated announcement: {announcement['message']}")
        
    except Exception as e:
        logger.error(f"Simulate vision error: {e}")
        emit('error', {'message': 'Vision simulation failed'})

@app.route('/mobile-demo')
def mobile_demo_interface():
    """Compatible mobile demo interface"""
    return render_template('mobile_compatible.html')
    

# Global variable for YAMNet model
yamnet_model = None
            
def classify_audio_with_yamnet(audio_data, sample_rate):
    """YAMNet pre-trained audio classification"""
    global yamnet_model
    
    try:
        if yamnet_model is None:
            logger.info("Loading YAMNet model...")
            yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            logger.info("YAMNet loaded successfully")
        
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        
        min_length = int(0.975 * 16000)
        if len(audio_data) < min_length:
            audio_data = np.pad(audio_data, (0, min_length - len(audio_data)))
        

        # Run YAMNet inference
        scores, embeddings, spectrogram = yamnet_model(audio_data)
        mean_scores = tf.reduce_mean(scores, axis=0)
        top_class_index = tf.argmax(mean_scores)
        confidence = tf.reduce_max(mean_scores).numpy()
        
        # Get class names using hardcoded list
        try:
            class_names = get_yamnet_class_names()
            if top_class_index < len(class_names):
                top_class_name = class_names[top_class_index]
            else:
                # Handle cases where index is beyond our list
                top_class_name = f"Audio_Class_{top_class_index}"
        except Exception as class_error:
            logger.warning(f"Could not get class names: {class_error}")
            top_class_name = f"Class_{top_class_index}"

  
        # Map YAMNet classes to our event types
        event_mapping = {
            'fire': 'fire_alarm',
            'alarm': 'fire_alarm',
            'smoke': 'fire_alarm',
            'siren': 'fire_alarm',
            'glass': 'glass_break',
            'shatter': 'glass_break',
            'breaking': 'glass_break',
            'crash': 'glass_break',
            'baby': 'baby_cry',
            'cry': 'baby_cry',
            'infant': 'baby_cry',
            'wail': 'baby_cry',
            'doorbell': 'doorbell',
            'bell': 'doorbell',
            'ding': 'doorbell',
            'chime': 'doorbell',
            'gunshot': 'gunshot',
            'gun': 'gunshot',
            'explosion': 'gunshot',
            'blast': 'gunshot',
            'silence': 'background_noise',
            'noise': 'background_noise'
        }
        
        # Find matching event type
        detected_event = 'background_noise'
        top_class_lower = str(top_class_name).lower()
        
        for keyword, our_event in event_mapping.items():
            if keyword in top_class_lower:
                detected_event = our_event
                break
        
        # Enhanced classification based on confidence and patterns
        if detected_event == 'background_noise':
            if confidence > 0.7:
                if any(word in top_class_lower for word in ['speech', 'voice', 'talk', 'human', 'male', 'female']):
                    detected_event = 'baby_cry'
                elif any(word in top_class_lower for word in ['music', 'sound', 'tone', 'note']):
                    detected_event = 'doorbell'
                elif any(word in top_class_lower for word in ['loud', 'bang', 'clap', 'snap', 'pop', 'thump']):
                    detected_event = 'fire_alarm'
                elif any(word in top_class_lower for word in ['whistle', 'beep', 'chirp', 'ding']):
                    detected_event = 'doorbell'
            elif confidence > 0.5:
                if any(word in top_class_lower for word in ['tap', 'knock', 'click']):
                    detected_event = 'doorbell'
                detected_event = 'fire_alarm'
        
        return detected_event, float(confidence), str(top_class_name)
        
        
    except Exception as e:
        logger.error(f"YAMNet error: {e}")
        # Enhanced fallback analysis
        rms_energy = np.sqrt(np.mean(audio_data**2))
        max_amplitude = np.max(np.abs(audio_data))
        
        if max_amplitude > 0.7:
            return 'fire_alarm', 0.8, 'Fallback_VeryLoud'
        elif max_amplitude > 0.4:
            return 'doorbell', 0.6, 'Fallback_Moderate' 
        elif max_amplitude > 0.1:
            return 'baby_cry', 0.5, 'Fallback_Quiet'
        else:
            return 'background_noise', 0.2, 'Fallback_Silent'
            

def get_yamnet_class_names():
    """Fallback YAMNet class names"""
    return [
        'Speech', 'Male_speech', 'Female_speech', 'Child_speech', 'Conversation',
        'Narration', 'Babbling', 'Speech_synthesizer', 'Shout', 'Bellow',
        'Whoop', 'Yell', 'Battle_cry', 'Children_shouting', 'Screaming',
        'Whispering', 'Laughter', 'Baby_laughter', 'Giggle', 'Snicker',
        'Belly_laugh', 'Chuckle', 'Chortle', 'Crying_sobbing', 'Baby_cry_infant_cry',
        'Whimper', 'Wail_moan', 'Sigh', 'Singing', 'Choir',
        'Yodeling', 'Chant', 'Mantra', 'Male_singing', 'Female_singing',
        'Child_singing', 'Synthetic_singing', 'Rapping', 'Humming', 'Groan',
        'Grunt', 'Whistling', 'Breathing', 'Wheeze', 'Snoring',
        'Gasp', 'Pant', 'Snort', 'Cough', 'Throat_clearing',
        'Sneeze', 'Sniff', 'Run', 'Shuffle', 'Walk_footsteps',
        'Chewing_mastication', 'Biting', 'Gargling', 'Stomach_rumble', 'Burping_eructation',
        'Hiccup', 'Fart', 'Hands', 'Finger_snapping', 'Clapping',
        'Heart_sounds_heartbeat', 'Heart_murmur', 'Cheering', 'Applause', 'Chatter',
        'Crowd', 'Hubbub_speech_noise_speech_babble', 'Children_playing', 'Animal', 'Domestic_animals_pets',
        'Dog', 'Bark', 'Yip', 'Howl', 'Bow-wow',
        'Growling', 'Whimper_dog', 'Cat', 'Purr', 'Meow',
        'Hiss', 'Caterwaul', 'Livestock_farm_animals_working_animals', 'Horse', 'Clip-clop',
        'Neigh_whinny', 'Cattle_bovine', 'Moo', 'Cowbell', 'Pig',
        'Oink', 'Goat', 'Bleat', 'Sheep', 'Fowl',
        'Chicken_rooster', 'Cluck', 'Crowing_cock-a-doodle-doo', 'Turkey', 'Gobble',
        'Duck', 'Quack', 'Goose', 'Honk', 'Wild_animals',
        'Roaring_cats_lions_tigers', 'Roar', 'Bird', 'Bird_vocalization_bird_call_bird_song',
        'Chirp_tweet', 'Squawk', 'Pigeon_dove', 'Coo', 'Crow',
        'Caw', 'Owl', 'Hoot', 'Bird_flight_flapping_wings', 'Canidae_dogs_wolves',
        'Rodents_rats_mice', 'Mouse', 'Patter', 'Insect', 'Cricket',
        'Mosquito', 'Fly_housefly', 'Buzz', 'Bee_wasp_etc', 'Frog',
        'Croak', 'Snake', 'Rattle', 'Whale_vocalization', 'Music',
        'Musical_instrument', 'Plucked_string_instrument', 'Guitar', 'Electric_guitar', 'Bass_guitar',
        'Acoustic_guitar', 'Steel_guitar_slide_guitar', 'Tapping_guitar_technique', 'Strum', 'Banjo',
        'Sitar', 'Mandolin', 'Zither', 'Ukulele', 'Keyboard_musical',
        'Piano', 'Electric_piano', 'Organ', 'Electronic_organ', 'Hammond_organ',
        'Synthesizer', 'Sampler', 'Harpsichord', 'Percussion', 'Timpani',
        'Tablas', 'Wood_block', 'Snare_drum', 'Drum_kit', 'Drum',
        'Cymbal', 'Hi-hat', 'Drum_machine', 'Tambourine', 'Rattle_instrument',
        'Maraca', 'Gong', 'Tubular_bells', 'Mallet_percussion', 'Marimba_xylophone',
        'Glockenspiel', 'Vibraphone', 'Steelpan', 'Orchestra', 'Brass_instrument',
        'French_horn', 'Trumpet', 'Trombone', 'Bowed_string_instrument', 'String_section',
        'Violin_fiddle', 'Pizzicato', 'Cello', 'Double_bass', 'Wind_instrument_woodwind_instrument',
        'Flute', 'Saxophone', 'Clarinet', 'Harp', 'Bell',
        'Church_bell', 'Jingle_bell', 'Bicycle_bell', 'Tuning_fork', 'Chime',
        'Wind_chime', 'Change_ringing_campanology', 'Harmonica', 'Accordion', 'Bagpipes',
        'Didgeridoo', 'Shofar', 'Theremin', 'Singing_bowl', 'Scratching_performance_technique',
        'Pop_music', 'Hip_hop_music', 'Beatboxing', 'Rock_music', 'Heavy_metal',
        'Punk_rock', 'Grunge', 'Progressive_rock', 'Rock_and_roll', 'Psychedelic_rock',
        'Rhythm_and_blues', 'Soul_music', 'Reggae', 'Country', 'Swing_music',
        'Bluegrass', 'Funk', 'Folk_music', 'Middle_Eastern_music', 'Jazz',
        'Disco', 'Classical_music', 'Opera', 'Electronic_music', 'House_music',
        'Techno', 'Dubstep', 'Drum_and_bass', 'Electronica', 'Electronic_dance_music',
        'Ambient_music', 'Trance_music', 'Music_of_Latin_America', 'Salsa_music', 'Flamenco',
        'Blues', 'Music_for_children', 'New-age_music', 'Vocal_music', 'A_capella',
        'Music_of_Africa', 'Afrobeat', 'Christian_music', 'Gospel_music', 'Music_of_Asia',
        'Carnatic_music', 'Music_of_Bollywood', 'Ska', 'Traditional_music', 'Independent_music',
        'Song', 'Background_music', 'Theme_music', 'Jingle_music', 'Soundtrack_music',
        'Lullaby', 'Video_game_music', 'Christmas_music', 'Dance_music', 'Wedding_music',
        'Happy_music', 'Funny_music', 'Sad_music', 'Tender_music', 'Exciting_music',
        'Angry_music', 'Scary_music', 'Wind', 'Rustling_leaves', 'Wind_noise_microphone',
        'Thunderstorm', 'Thunder', 'Water', 'Rain', 'Raindrop',
        'Rain_on_surface', 'Stream', 'Waterfall', 'Ocean', 'Waves_surf',
        'Steam', 'Gurgling', 'Fire', 'Crackle', 'Vehicle',
        'Boat_Water_vehicle', 'Sailboat_sailing_ship', 'Rowboat_canoe_kayak', 'Motorboat_speedboat', 'Ship',
        'Motor_vehicle_road', 'Car', 'Vehicle_horn_car_horn_honking', 'Toot', 'Car_alarm',
        'Power_windows_electric_windows', 'Skidding', 'Tire_squeal', 'Car_passing_by', 'Race_car_auto_racing',
        'Truck', 'Air_brake', 'Air_horn_truck_horn', 'Reversing_beeps', 'Ice_cream_truck_ice_cream_van',
        'Bus', 'Emergency_vehicle', 'Police_car_siren', 'Ambulance_siren', 'Fire_engine_fire_truck_siren',
        'Motorcycle', 'Traffic_noise_roadway_noise', 'Rail_transport', 'Train', 'Train_whistle',
        'Train_horn', 'Railroad_car_train_wagon', 'Train_wheels_squealing', 'Subway_metro_underground',
        'Aircraft', 'Aircraft_engine', 'Jet_engine', 'Propeller_airscrew', 'Helicopter',
        'Fixed-wing_aircraft_airplane', 'Bicycle', 'Skateboard', 'Engine', 'Light_engine_high_frequency',
        'Dental_drill_dentists_drill', 'Lawn_mower', 'Chainsaw', 'Medium_engine_mid_frequency', 'Heavy_engine_low_frequency',
        'Engine_accelerating_revving_vroom', 'Engine_starting', 'Idling', 'Engine_knocking', 'Tools',
        'Hammer', 'Wood', 'Sawing', 'Filing_rasp', 'Sanding',
        'Power_tool', 'Drill', 'Jackhammer', 'Sawmill', 'Gunshot_gunfire',
        'Machine_gun', 'Fusillade', 'Artillery_fire', 'Cap_gun', 'Fireworks',
        'Firecracker', 'Burst_pop', 'Eruption', 'Boom', 'Generic_impact_sounds',
        'Crash', 'Breaking', 'Bouncing', 'Whip', 'Slam',
        'Thump_thud', 'Smash_crash', 'Bang', 'Tap', 'Knock',
        'Slap_smack', 'Whack_thwack', 'Punch', 'Wood', 'Chop',
        'Splinter', 'Crack', 'Glass', 'Chink_clink', 'Shatter',
        'Liquid', 'Splash_splatter', 'Slosh', 'Squish', 'Drip',
        'Pour', 'Trickle_dribble', 'Gush', 'Fill_with_liquid', 'Spray',
        'Pump_liquid', 'Stir', 'Boiling', 'Sonar', 'Arrow',
        'Whoosh_swoosh_swish', 'Thunk', 'Electronic_tuner', 'Effects_unit', 'Chorus_effect',
        'Basketball_bounce', 'Bang', 'Honk', 'Toot', 'Beep_bleep',
        'Ping', 'Ding', 'Clang', 'Squeak', 'Rustle',
        'Whir', 'Clatter', 'Sizzle', 'Clicking', 'Tick',
        'Beeping', 'Cash_register', 'Sine_wave', 'Chirp_tone', 'Sound_effect',
        'Pulse', 'Electronic_music', 'Scary_music', 'Lullaby', 'Buzz',
        'Electric_shaver_electric_razor', 'Toothbrush', 'Vacuum_cleaner', 'Zipper_clothing', 'Keys_jangling',
        'Coin_dropping', 'Scissors', 'Electric_toothbrush', 'Mechanical_fan', 'Air_conditioning',
        'Microwave_oven', 'Blender', 'Water_tap_faucet', 'Sink_filling_or_washing', 'Bathtub_filling_or_washing',
        'Hair_dryer', 'Toilet_flush', 'Thunderstorm', 'Silence'
    ]

# Global variable for MobileNet
mobilenet_model = None

def classify_image_with_pretrained_models(image):
    """Classify image using MobileNet and OpenCV face detection"""
    global mobilenet_model
    
    try:
        height, width = image.shape[:2]
        
        # 1. Face detection using OpenCV (pre-trained)
        face_confidence, num_faces = detect_faces_opencv(image)
        
        # 2. Object classification using MobileNet (pre-trained)
        top_objects = classify_with_mobilenet(image)
        
        # 3. Basic image analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        detection_details = f"Faces:{num_faces}, Objects:{top_objects[:2]}, Brightness:{brightness:.1f}"
        
        # Decision logic based on pre-trained model results
        if num_faces > 0:
            if num_faces > 2:
                return 'crowd_density', min(0.9, face_confidence + 0.2), f"Multiple faces: {num_faces}"
            else:
                return 'unknown_person', face_confidence, f"Single face detected"
        elif brightness < 30:
            return 'low_light', 0.75, f"Low brightness: {brightness:.1f}"
        elif any('person' in obj.lower() for obj in top_objects):
            return 'unknown_person', 0.70, f"Person detected in objects: {top_objects[0]}"
        elif len(top_objects) > 0 and any(word in top_objects[0].lower() for word in ['bag', 'box', 'container', 'suitcase']):
            return 'suspicious_object', 0.65, f"Suspicious object: {top_objects[0]}"
        else:
            return 'normal', 0.25, "No significant features"
            
    except Exception as e:
        logger.error(f"Vision classification error: {e}")
        return 'normal', 0.1, f"Error: {str(e)}"

def detect_faces_opencv(image):
    """Face detection using OpenCV Haar cascades (pre-trained)"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        confidence = 0.8 if len(faces) > 0 else 0.1
        return confidence, len(faces)
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return 0.1, 0

def classify_with_mobilenet(image):
    """Object classification using MobileNet (pre-trained)"""
    global mobilenet_model
    
    try:
        if mobilenet_model is None:
            logger.info("Loading MobileNet model...")
            mobilenet_model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=True,
                input_shape=(224, 224, 3)
            )
            logger.info("MobileNet loaded successfully")
        
        # Preprocess for MobileNet
        resized = cv2.resize(image, (224, 224))
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = tf.keras.applications.mobilenet_v2.preprocess_input(rgb_image)
        batch = np.expand_dims(normalized, axis=0)
        
        # Run inference
        predictions = mobilenet_model.predict(batch, verbose=0)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        
        # Extract top object names
        object_names = [pred[1] for pred in decoded if pred[2] > 0.1]
        
        return object_names if object_names else ['unknown']
        
    except Exception as e:
        logger.error(f"MobileNet error: {e}")
        return ['error'] 
    
if __name__ == '__main__':
    # Initialize system
    if initialize_system():
        logger.info("Pre-trained model system initialized successfully")
        
        # Get server configuration
        server_config = config.get('server', {})
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 5000)
        debug = server_config.get('debug', False)
        
        # Reduce logging noise - set to WARNING to hide HTTP requests
        import logging
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('engineio').setLevel(logging.WARNING)
        logging.getLogger('socketio').setLevel(logging.WARNING)
        
        # Start server with reduced debug output
        socketio.run(app, host=host, port=port, debug=False, log_output=False)
    else:
        print("Failed to initialize system. Exiting...")










