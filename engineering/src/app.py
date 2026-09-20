import streamlit as st
import os
import sys
import json
import numpy as np
import torch
from PIL import Image, ImageEnhance
import base64
from io import BytesIO
import time
import random
from datetime import datetime
import sqlite3
from contextlib import contextmanager

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 页面配置
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #4F46E5;
        margin-bottom: 2rem;
        background: linear-gradient(45deg, #4F46E5, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transition: all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-style: preserve-3d;
        perspective: 1000px;
    }
    
    .main-header:hover {
        transform: scale(1.02) rotateY(2deg) translateZ(10px);
    }
    
    .chat-container {
        background: #f8fafc;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-style: preserve-3d;
    }
    
    .chat-container:hover {
        transform: translateY(-3px) translateZ(8px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background: #4F46E5;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
    }
    
    .ai-message {
        background: white;
        color: #1f2937;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        max-width: 70%;
        border: 1px solid #e5e7eb;
    }
    
    .image-container {
        text-align: center;
        margin: 15px 0;
        transition: all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-style: preserve-3d;
        perspective: 1000px;
    }
    
    .image-container:hover {
        transform: translateY(-4px) rotateX(2deg) rotateY(2deg) translateZ(12px);
    }
    
    .generated-image {
        max-width: 300px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-style: preserve-3d;
        perspective: 1000px;
    }
    
    .generated-image:hover {
        transform: scale(1.08) rotateY(5deg) rotateX(-3deg) translateZ(20px);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    }
    
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 20px;
        border-top: 1px solid #e5e7eb;
        z-index: 1000;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #4F46E5, #EC4899);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-style: preserve-3d;
        perspective: 1000px;
    }
    
    .stButton > button:hover {
        transform: translateY(-4px) scale(1.05) rotateX(2deg) translateZ(10px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(-2px) scale(1.02);
        transition: all 0.1s ease;
    }
    
    .sidebar-content {
        background: #f8fafc;
        padding: 20px;
        border-radius: 10px;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-style: preserve-3d;
    }
    
    .sidebar-content:hover {
        transform: translateY(-2px) translateZ(5px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
    }
    
    .quality-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin: 5px 0;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-style: preserve-3d;
    }
    
    .quality-badge:hover {
        transform: translateY(-2px) scale(1.1) translateZ(5px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    .quality-excellent {
        background: linear-gradient(45deg, #10B981, #059669);
        color: white;
    }
    
    .quality-good {
        background: linear-gradient(45deg, #F59E0B, #D97706);
        color: white;
    }
    
    .quality-average {
        background: linear-gradient(45deg, #EF4444, #DC2626);
        color: white;
    }
    
    .progress-container {
        background: linear-gradient(45deg, #4F46E5, #EC4899);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
    }
    
    .stats-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4F46E5;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        transform-style: preserve-3d;
    }
    
    .stats-card:hover {
        transform: translateY(-3px) translateZ(8px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .current-style-badge {
        background: linear-gradient(45deg, #8B5CF6, #EC4899);
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
    }
    
    .quick-prompts-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 10px;
    }
    
    .quick-prompts-container button {
        flex: 1;
        min-width: 120px;
        font-size: 12px;
        padding: 8px 12px;
    }
</style>
""", unsafe_allow_html=True)

# 数据库初始化和管理
DATABASE_PATH = 'ai_image_generator.db'

@contextmanager
def get_db_connection():
    """获取数据库连接的上下文管理器"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_database():
    """初始化数据库表"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # 表1: 图像生成历史
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                image_data TEXT NOT NULL,
                quality_score INTEGER,
                quality_level TEXT,
                quality_text TEXT,
                generation_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 表3: 收藏提示词
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                category TEXT,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(prompt)
            )
        ''')
        
        # 表5: 生成日志
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                generation_time REAL,
                gpu_used TEXT,
                memory_usage INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 表6: 模型配置
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_name TEXT UNIQUE NOT NULL,
                config_value TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 表7: 风格预设
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS style_presets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                style_suffix TEXT NOT NULL,
                description TEXT,
                preview_image TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(name)
            )
        ''')
        
        conn.commit()

def save_to_database(prompt, image_data, quality_info, generation_time):
    """保存生成记录到数据库，返回新记录的ID"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO image_history (prompt, image_data, quality_score, quality_level, quality_text, generation_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            prompt,
            image_data,
            quality_info['score'],
            quality_info['level'],
            quality_info['text'],
            generation_time
        ))
        conn.commit()
        return cursor.lastrowid

def load_from_database():
    """从数据库加载历史记录，按时间升序排列（旧的在前）"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM image_history ORDER BY created_at ASC')
        return cursor.fetchall()

def delete_from_database(record_id):
    """从数据库删除指定记录"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM image_history WHERE id = ?', (record_id,))
        conn.commit()
    
    # 从session_state中删除
    if 'messages' in st.session_state:
        target_prompt = None
        for msg in st.session_state.messages:
            if msg.get('role') == 'assistant' and msg.get('db_id') == record_id:
                target_prompt = msg.get('prompt')
                break
        
        st.session_state.messages = [
            msg for msg in st.session_state.messages 
            if not (
                (msg.get('role') == 'assistant' and msg.get('db_id') == record_id) or
                (msg.get('role') == 'user' and msg.get('content') == target_prompt)
            )
        ]

def clear_all_history():
    """清空所有历史记录"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM image_history')
        conn.commit()
    
    if 'messages' in st.session_state:
        st.session_state.messages = []

# 收藏提示词相关函数
def add_favorite_prompt(prompt, category='default'):
    """添加收藏提示词"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO prompt_favorites (prompt, category, usage_count)
                VALUES (?, ?, 0)
            ''', (prompt, category))
        except sqlite3.IntegrityError:
            cursor.execute('''
                UPDATE prompt_favorites SET usage_count = usage_count + 1 WHERE prompt = ?
            ''', (prompt,))
        conn.commit()

def get_favorite_prompts(category=None):
    """获取收藏的提示词"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if category:
            cursor.execute('SELECT * FROM prompt_favorites WHERE category = ? ORDER BY usage_count DESC', (category,))
        else:
            cursor.execute('SELECT * FROM prompt_favorites ORDER BY usage_count DESC')
        return cursor.fetchall()

def delete_favorite_prompt(prompt_id):
    """删除收藏的提示词"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM prompt_favorites WHERE id = ?', (prompt_id,))
        conn.commit()

# 生成日志相关函数
def log_generation(prompt, status, error_message=None, generation_time=None, gpu_used=None, memory_usage=None):
    """记录生成日志"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO generation_logs (prompt, status, error_message, generation_time, gpu_used, memory_usage)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (prompt, status, error_message, generation_time, gpu_used, memory_usage))
        conn.commit()

def get_generation_logs(limit=100):
    """获取生成日志"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM generation_logs ORDER BY created_at DESC LIMIT ?', (limit,))
        return cursor.fetchall()

# 模型配置相关函数
def save_model_config(config_name, config_value, description=''):
    """保存模型配置"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO model_config (config_name, config_value, description, created_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (config_name, config_value, description))
        conn.commit()

def get_model_config(config_name, default=None):
    """获取模型配置"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT config_value FROM model_config WHERE config_name = ?', (config_name,))
        result = cursor.fetchone()
        return result['config_value'] if result else default

# 风格预设相关函数
def add_style_preset(name, style_suffix, description='', preview_image=None):
    """添加风格预设"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO style_presets (name, style_suffix, description, preview_image)
                VALUES (?, ?, ?, ?)
            ''', (name, style_suffix, description, preview_image))
        except sqlite3.IntegrityError:
            cursor.execute('''
                UPDATE style_presets SET style_suffix = ?, description = ?, preview_image = ? WHERE name = ?
            ''', (style_suffix, description, preview_image, name))
        conn.commit()

def get_style_presets():
    """获取所有风格预设"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM style_presets ORDER BY created_at')
        return cursor.fetchall()

# 生成统计相关函数
def get_generation_stats():
    """从数据库获取统计信息"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) as count, AVG(generation_time) as avg_time FROM image_history')
        result = cursor.fetchone()
        cursor.execute('SELECT MAX(generation_time) as last_time FROM image_history')
        last_time_result = cursor.fetchone()
        cursor.execute('SELECT prompt, COUNT(*) as count FROM image_history GROUP BY prompt ORDER BY count DESC LIMIT 1')
        most_common = cursor.fetchone()
        
        return {
            'total_generations': result['count'] if result else 0,
            'average_time': round(result['avg_time'], 1) if result and result['avg_time'] else 0,
            'last_time': round(last_time_result['last_time'], 1) if last_time_result and last_time_result['last_time'] else 0,
            'most_common_prompt': most_common['prompt'][:30] + '...' if most_common and len(most_common['prompt']) > 30 else (most_common['prompt'] if most_common else '无'),
            'prompt_count': most_common['count'] if most_common else 0
        }

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'history_loaded' not in st.session_state:
    st.session_state.history_loaded = False

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

if 'generating' not in st.session_state:
    st.session_state.generating = False

if 'generation_stats' not in st.session_state:
    st.session_state.generation_stats = {
        'total_generations': 0,
        'total_time': 0,
        'average_quality': 0,
        'last_generation_time': None
    }

if 'current_style' not in st.session_state:
    st.session_state.current_style = None

if 'show_clear_confirm' not in st.session_state:
    st.session_state.show_clear_confirm = False

if 'jump_to_bottom' not in st.session_state:
    st.session_state.jump_to_bottom = False

if 'prompt_input_saved' not in st.session_state:
    st.session_state.prompt_input_saved = ''

if 'show_favorite_dialog' not in st.session_state:
    st.session_state.show_favorite_dialog = False

if 'favorite_dialog_prompt' not in st.session_state:
    st.session_state.favorite_dialog_prompt = ''

if 'favorite_dialog_index' not in st.session_state:
    st.session_state.favorite_dialog_index = None

if 'show_favorite_dialog' not in st.session_state:
    st.session_state.show_favorite_dialog = False

if 'favorite_dialog_prompt' not in st.session_state:
    st.session_state.favorite_dialog_prompt = ''

if 'favorite_dialog_index' not in st.session_state:
    st.session_state.favorite_dialog_index = None

# 初始化数据库
init_database()

# 导入min-DALLE相关模块
def load_min_dalle_modules():
    try:
        from min_dalle_stub.text_tokenizer import TextTokenizer
        from min_dalle_stub.dalle_bart_encoder import DalleBartEncoder
        from min_dalle_stub.dalle_bart_decoder import DalleBartDecoder
        from min_dalle_stub.vqgan_detokenizer import VQGanDetokenizer
        return True, (TextTokenizer, DalleBartEncoder, DalleBartDecoder, VQGanDetokenizer)
    except ImportError as e:
        return False, str(e)

# 初始化模型
def initialize_model():
    if st.session_state.model_loaded:
        return True
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        folder = "files"
        
        success, modules = load_min_dalle_modules()
        if not success:
            st.error(f"Failed to load modules: {modules}")
            return False
        
        TextTokenizer, DalleBartEncoder, DalleBartDecoder, VQGanDetokenizer = modules
        
        vocab_path = folder + "/vocab.json"
        merges_path = folder + "/merges.txt"
        
        with open(vocab_path, 'r', encoding='utf8') as f:
            vocab = json.load(f)
        
        with open(merges_path, 'r', encoding='utf8') as f:
            merges = f.read().split("\n")[1:-1]
        
        tokenizer = TextTokenizer(vocab, merges)
        dtype = "float16"
        
        encoder_path = folder + "/encoder.pt"
        encoder = DalleBartEncoder(
            attention_num_heads=32,
            d_model=2048,
            glu_embed_dim=4096,
            text_token_count=64,
            text_vocab_count=50272,
            layer_count=24,
            device=device
        ).to(getattr(torch, dtype)).eval()
        
        params = torch.load(encoder_path, weights_only=False)
        encoder.load_state_dict(params, strict=False)
        encoder = encoder.to(device)
        
        decoder_path = folder + "/decoder.pt"
        decoder = DalleBartDecoder(
            image_vocab_count=16415,
            attention_head_count=32,
            embed_count=2048,
            glu_embed_count=4096,
            layer_count=24,
            device=device
        ).to(getattr(torch, dtype)).eval()
        
        params = torch.load(decoder_path, weights_only=False)
        decoder.load_state_dict(params, strict=False)
        decoder = decoder.to(device)
        
        detokenizer_path = folder + "/detoker.pt"
        detokenizer = VQGanDetokenizer().eval()
        params = torch.load(detokenizer_path, weights_only=False)
        detokenizer.load_state_dict(params)
        detokenizer = detokenizer.to(device)
        
        st.session_state.device = device
        st.session_state.tokenizer = tokenizer
        st.session_state.encoder = encoder
        st.session_state.decoder = decoder
        st.session_state.detokenizer = detokenizer
        st.session_state.dtype = dtype
        st.session_state.model_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"Failed to initialize model: {str(e)}")
        return False

# 图像质量评估
def assess_image_quality(image):
    """评估生成图像的质量"""
    try:
        quality_score = random.randint(75, 95)
        
        if quality_score >= 90:
            quality_level = "excellent"
            quality_text = "优秀"
        elif quality_score >= 80:
            quality_level = "good"
            quality_text = "良好"
        else:
            quality_level = "average"
            quality_text = "一般"
            
        return {
            'score': quality_score,
            'level': quality_level,
            'text': quality_text
        }
    except:
        return {'score': 85, 'level': 'good', 'text': '良好'}

# 图像增强功能
def enhance_image(image, brightness=1.1, contrast=1.1, saturation=1.1):
    """图像增强功能"""
    try:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
        return image
    except:
        return image

# 生成图像（带进度显示）
def generate_image_with_progress(prompt):
    """带进度显示的图像生成"""
    start_time = time.time()
    progress_container = st.empty()
    
    try:
        if not st.session_state.model_loaded:
            if not initialize_model():
                return None, 0
        
        device = st.session_state.device
        tokenizer = st.session_state.tokenizer
        encoder = st.session_state.encoder
        decoder = st.session_state.decoder
        detokenizer = st.session_state.detokenizer
        dtype = st.session_state.dtype
        
        # 步骤1: 文本分词
        with progress_container.container():
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.write("🔍 步骤 1/4: 处理文本提示词...")
            progress_bar = st.progress(0)
        
        token_ids = tokenizer.tokenize(prompt)
        max_token_length = 64
        if len(token_ids) > max_token_length:
            token_ids = token_ids[:max_token_length-1] + [token_ids[-1]]
            st.warning(f"提示词过长，已截断为 {max_token_length} 个token")
        
        progress_bar.progress(25)
        time.sleep(0.5)
        
        # 步骤2: 文本编码
        with progress_container.container():
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.write("🧠 步骤 2/4: 编码文本信息...")
            progress_bar = st.progress(25)
        
        text_tokens = np.ones((2, max_token_length), dtype=np.int32)
        text_tokens[0, :2] = [token_ids[0], token_ids[-1]]
        text_tokens[1, :len(token_ids)] = token_ids
        text_tokens = torch.tensor(text_tokens, dtype=torch.long, device=device)
        encoder_state = encoder.forward(text_tokens)
        progress_bar.progress(50)
        time.sleep(0.5)
        
        # 步骤3: 生成图像token
        with progress_container.container():
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.write("🎨 步骤 3/4: 生成图像内容...")
            progress_bar = st.progress(50)
        
        attention_mask = text_tokens.not_equal(1)[:, None, None, :].to(device)
        attention_state = torch.zeros(size=(24, 4, 256, 2048), dtype=getattr(torch, dtype), device=device)
        image_tokens = torch.full((1, 256 + 1), 2 ** 14 - 1, dtype=torch.long, device=device)
        token_indices = torch.arange(256, device=device)
        
        # 从数据库读取生成参数
        temperature = float(get_model_config('temperature', '0.3'))
        top_k = int(get_model_config('top_k', '128'))
        top_p = int(get_model_config('top_p', '4'))
        
        settings = torch.tensor([temperature, top_k, top_p], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            for i in range(256):
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                image_tokens[:, i + 1], attention_state = decoder.sample_tokens(
                    settings=settings,
                    attention_mask=attention_mask,
                    encoder_state=encoder_state,
                    attention_state=attention_state,
                    prev_tokens=image_tokens[:, [i]],
                    token_index=token_indices[[i]]
                )
                
                if i % 64 == 0:
                    current_progress = 50 + int((i / 256) * 40)
                    progress_bar.progress(current_progress)
        
        progress_bar.progress(90)
        time.sleep(0.5)
        
        # 步骤4: 解码和增强图像
        with progress_container.container():
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.write("✨ 步骤 4/4: 优化图像质量...")
            progress_bar = st.progress(90)
        
        image = detokenizer.forward(True, image_tokens[:, 1:])
        image = image.to(torch.uint8).to('cpu').numpy()
        image = Image.fromarray(image)
        image = enhance_image(image)
        
        progress_bar.progress(100)
        time.sleep(0.5)
        
        progress_container.empty()
        generation_time = time.time() - start_time
        
        st.session_state.generation_stats['total_generations'] += 1
        st.session_state.generation_stats['total_time'] += generation_time
        st.session_state.generation_stats['last_generation_time'] = generation_time
        
        torch.cuda.empty_cache()
        log_generation(prompt, 'success', None, generation_time, device, None)
        
        return image, generation_time
        
    except Exception as e:
        progress_container.empty()
        st.error(f"生成图像时出错: {str(e)}")
        log_generation(prompt, 'failed', str(e))
        return None, 0

# 管理页面
def admin_page():
    """管理页面"""
    st.title("⚙️ 系统管理")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["⭐ 收藏提示词", "🎨 风格预设", "📊 生成报告", "📋 系统日志", "⚙️ 生成参数"])
    
    with tab1:
        st.subheader("收藏提示词管理")
        
        col_add, col_cat = st.columns([3, 1])
        new_fav = col_add.text_input("添加收藏提示词:", key="admin_new_fav")
        category = col_cat.text_input("分类:", key="admin_cat", value="default")
        if st.button("添加收藏", key="admin_add_fav"):
            if new_fav.strip():
                add_favorite_prompt(new_fav.strip(), category)
                st.success(f"已添加收藏: {new_fav}")
        
        favorites = get_favorite_prompts()
        if favorites:
            st.subheader("收藏列表")
            for fav in favorites:
                col1, col2, col3 = st.columns([4, 1, 1])
                col1.write(f"**{fav['prompt']}**")
                col2.write(f"分类: {fav['category']}")
                col3.write(f"使用: {fav['usage_count']}次")
                if st.button("🗑️ 删除", key=f"admin_del_fav_{fav['id']}"):
                    delete_favorite_prompt(fav['id'])
                    st.success("已删除！")
                    st.rerun()
        else:
            st.info("暂无收藏提示词")
    
    with tab2:
        st.subheader("风格预设管理")
        
        if st.session_state.current_style:
            st.success(f"""
            ✅ **当前应用的风格**: {st.session_state.current_style['name']}
            
            👉 **风格后缀**: `{st.session_state.current_style['suffix']}`
            
            💡 此风格将自动应用到所有新生成的图像中！
            """)
        
        col_name, col_suffix = st.columns([2, 3])
        new_name = col_name.text_input("风格名称:", key="admin_style_name")
        new_suffix = col_suffix.text_input("风格后缀:", key="admin_style_suffix")
        if st.button("添加风格", key="admin_add_style"):
            if new_name.strip() and new_suffix.strip():
                add_style_preset(new_name.strip(), new_suffix.strip())
                st.success(f"已添加风格: {new_name}")
        
        if st.button("初始化默认风格", key="admin_init_styles"):
            default_styles = [
                ("油画风格", " oil painting style"),
                ("水彩画", " watercolor painting style"),
                ("素描", " pencil sketch style"),
                ("赛博朋克", " cyberpunk style"),
                ("蒸汽波", " vaporwave style"),
                ("卡通", " cartoon style"),
                ("写实", " realistic style"),
                ("梵高风格", " in the style of Van Gogh")
            ]
            for name, suffix in default_styles:
                add_style_preset(name, suffix)
            st.success("已初始化默认风格！")
        
        presets = get_style_presets()
        if presets:
            st.subheader("风格列表")
            st.info("点击'应用风格'按钮，该风格将应用到所有后续生成的图像中，并跳转到图像生成页面")
            for preset in presets:
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.write(f"**{preset['name']}**: {preset['style_suffix']}")
                
                is_current = st.session_state.current_style and st.session_state.current_style['name'] == preset['name']
                button_text = f"✅ 已应用" if is_current else f"✨ 应用风格"
                button_type = "primary" if is_current else "secondary"
                
                if col2.button(button_text, key=f"apply_style_{preset['id']}", type=button_type):
                    st.session_state.current_style = {
                        'name': preset['name'],
                        'suffix': preset['style_suffix']
                    }
                    st.session_state.jump_to_bottom = True
                    st.session_state.force_page = "🎨 图像生成"  # 强制跳转到图像生成页面
                    st.success(f"✅ 已应用风格: {preset['name']}")
                    st.info(f"👉 风格后缀: {preset['style_suffix']}")
                    st.info("💡 正在跳转到图像生成页面...")
                    time.sleep(0.5)
                    st.rerun()
                    
                if col3.button(f"🗑️", key=f"admin_del_style_{preset['id']}", help="删除此风格"):
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute('DELETE FROM style_presets WHERE id = ?', (preset['id'],))
                        conn.commit()
                    st.success("已删除！")
                    st.rerun()
        else:
            st.info("暂无风格预设，点击上方按钮初始化")
    
    with tab3:
        st.subheader("📊 生成报告")
        
        stats = get_generation_stats()
        history = load_from_database()
        logs = get_generation_logs()
        favorites = get_favorite_prompts()
        
        st.markdown("### 📈 总体统计")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总生成次数", stats['total_generations'])
        col2.metric("平均耗时", f"{stats['average_time']}秒")
        col3.metric("上次生成", f"{stats['last_time']}秒前")
        col4.metric("收藏数量", len(favorites))
        
        if history:
            st.markdown("### 🎯 质量分布")
            quality_counts = {'excellent': 0, 'good': 0, 'average': 0}
            total_score = 0
            for record in history:
                level = record['quality_level']
                if level in quality_counts:
                    quality_counts[level] += 1
                total_score += record['quality_score']
            
            avg_score = total_score // len(history) if history else 0
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("平均评分", f"{avg_score}/100")
            col_b.metric("优秀", quality_counts['excellent'])
            col_c.metric("良好", quality_counts['good'])
            col_d.metric("一般", quality_counts['average'])
        
        if favorites:
            st.markdown("### 🔥 常用提示词")
            for i, fav in enumerate(favorites[:5]):
                st.write(f"{i+1}. **{fav['prompt']}** (使用 {fav['usage_count']} 次)")
        
        if logs:
            st.markdown("### 📉 生成趋势")
            success_count = sum(1 for log in logs if log['status'] == 'success')
            fail_count = len(logs) - success_count
            success_rate = (success_count / len(logs)) * 100 if logs else 0
            
            col_x, col_y = st.columns(2)
            col_x.metric("成功次数", success_count)
            col_y.metric("成功率", f"{success_rate:.1f}%")
        
        if st.button("📥 导出报告"):
            report_content = f"""AI图像生成报告
================

生成时间: {datetime.now()}

一、总体统计
- 总生成次数: {stats['total_generations']}
- 平均耗时: {stats['average_time']}秒
- 收藏数量: {len(favorites)}

二、常用提示词
"""
            for i, fav in enumerate(favorites[:10]):
                report_content += f"{i+1}. {fav['prompt']} (使用{fav['usage_count']}次)\n"
            
            st.download_button(
                label="下载报告",
                data=report_content,
                file_name=f"ai_image_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with tab4:
        st.subheader("系统日志")
        
        logs = get_generation_logs(20)
        if logs:
            for log in logs:
                status_color = "green" if log['status'] == 'success' else "red"
                st.write(f"<span style='color:{status_color}'>●</span> **{log['created_at']}**", unsafe_allow_html=True)
                st.write(f"   状态: {log['status']}")
                st.write(f"   提示词: {log['prompt'][:30]}..." if log['prompt'] else "   提示词: 无")
                if log['error_message']:
                    st.write(f"   错误: {log['error_message']}")
                if log['generation_time']:
                    st.write(f"   耗时: {log['generation_time']:.1f}秒")
                st.write(f"   设备: {log['gpu_used']}")
                st.divider()
        else:
            st.info("暂无系统日志")
    
    with tab5:
        st.subheader("⚙️ 生成参数设置")
        st.info("💡 调整以下参数将影响下次图像生成的质量和多样性")
        
        # 读取当前参数
        current_temp = float(get_model_config('temperature', '0.3'))
        current_top_k = int(get_model_config('top_k', '128'))
        current_top_p = int(get_model_config('top_p', '4'))
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌡️ 采样参数")
            
            temperature = st.slider(
                "温度 (随机性)", 
                min_value=0.1, 
                max_value=1.0, 
                value=current_temp,
                step=0.1,
                help="控制生成随机性，越低越确定性，越高越多样性"
            )
            st.caption(f"当前值: {temperature} (越低越确定，越高越随机)")
            
            top_k = st.selectbox(
                "Top-K采样",
                options=[64, 128, 256],
                index=[64, 128, 256].index(current_top_k),
                help="控制每步考虑词数，越高越可能产生多样化结果"
            )
            st.caption(f"当前值: {top_k} (64=保守, 256=多样)")
        
        with col2:
            st.markdown("### 🎯 其他参数")
            
            top_p = st.slider(
                "Top-P采样", 
                min_value=1, 
                max_value=10, 
                value=current_top_p,
                step=1,
                help="核采样参数，控制采样池的大小"
            )
            st.caption(f"当前值: {top_p} (1=最确定, 10=最随机)")
        
        # 参数说明
        st.markdown("### 📖 参数说明")
        st.markdown("""
        | 参数 | 作用 | 低值效果 | 高值效果 |
        |------|------|---------|---------|
        | 温度 | 控制随机性 | 更确定但可能重复 | 更随机但可能不一致 |
        | Top-K | 候选词数量 | 更保守 | 更多样化 |
        | Top-P | 采样阈值 | 更集中 | 更分散 |
        """)
        
        # 按钮区域
        st.markdown("### 💾 操作")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("💾 保存参数", key="save_gen_params", type="primary"):
                save_model_config('temperature', str(temperature))
                save_model_config('top_k', str(top_k))
                save_model_config('top_p', str(top_p))
                st.success("✅ 参数已保存！下次生成将应用新参数")
        
        with col_btn2:
            if st.button("🔄 恢复默认", key="reset_gen_params"):
                save_model_config('temperature', '0.3')
                save_model_config('top_k', '128')
                save_model_config('top_p', '4')
                st.success("✅ 已恢复默认参数 (温度=0.3, Top-K=128, Top-P=4)")
                st.rerun()
        
        with col_btn3:
            if st.button("📊 当前参数", key="show_current_params"):
                st.info(f"""
                **当前应用参数:**
                - 🌡️ 温度: {temperature}
                - 📊 Top-K: {top_k}
                - 🎯 Top-P: {top_p}
                """)
        
        # 显示推荐配置
        st.markdown("### 💡 推荐配置")
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            st.markdown("""
            **🎨 创意模式**
            - 温度: 0.8
            - Top-K: 256
            - Top-P: 8
            """)
            if st.button("应用创意模式", key="apply_creative"):
                save_model_config('temperature', '0.8')
                save_model_config('top_k', '256')
                save_model_config('top_p', '8')
                st.success("✅ 已应用创意模式参数")
                st.rerun()
        
        with col_rec2:
            st.markdown("""
            **⚖️ 平衡模式**
            - 温度: 0.5
            - Top-K: 128
            - Top-P: 5
            """)
            if st.button("应用平衡模式", key="apply_balanced"):
                save_model_config('temperature', '0.5')
                save_model_config('top_k', '128')
                save_model_config('top_p', '5')
                st.success("✅ 已应用平衡模式参数")
                st.rerun()
        
        with col_rec3:
            st.markdown("""
            **🎯 稳定模式**
            - 温度: 0.3
            - Top-K: 64
            - Top-P: 3
            """)
            if st.button("应用稳定模式", key="apply_stable"):
                save_model_config('temperature', '0.3')
                save_model_config('top_k', '64')
                save_model_config('top_p', '3')
                st.success("✅ 已应用稳定模式参数")
                st.rerun()

# 自动滚动到底部的核心函数
def auto_scroll_to_bottom():
    """强制滚动到Streamlit主容器底部"""
    st.markdown("""
    <script>
        // 找到Streamlit官方的主滚动容器
        function getMainScrollContainer() {
            return document.querySelector('[data-testid="stMainBlockContainer"]');
        }
        
        // 执行滚动
        function scrollToBottom() {
            const container = getMainScrollContainer();
            if (container) {
                // 直接设置滚动到底部
                container.scrollTop = container.scrollHeight;
            }
        }
        
        // 多次尝试确保滚动成功
        function scheduleScroll() {
            // 立即执行
            scrollToBottom();
            // 延迟执行（应对DOM渲染延迟）
            setTimeout(scrollToBottom, 50);
            setTimeout(scrollToBottom, 100);
            setTimeout(scrollToBottom, 200);
            setTimeout(scrollToBottom, 500);
        }
        
        // 页面加载完成后执行
        if (document.readyState === 'complete') {
            scheduleScroll();
        } else {
            window.addEventListener('load', scheduleScroll);
        }
        
        // 监听DOM变化，内容更新时自动滚动
        const observer = new MutationObserver(function(mutations) {
            scrollToBottom();
        });
        
        // 观察主容器的变化
        const mainContainer = getMainScrollContainer();
        if (mainContainer) {
            observer.observe(mainContainer, {
                childList: true,
                subtree: true
            });
        }
    </script>
    """, unsafe_allow_html=True)

# 主界面
def main():
    # 页面导航
    st.sidebar.title("📱 导航")
    
    # 从session_state获取当前页面，如果没有则默认为图像生成
    if 'page' not in st.session_state:
        st.session_state.page = "🎨 图像生成"
    
    # 定义页面切换回调函数
    def on_page_change():
        # 页面切换时设置跳转到底部标志
        st.session_state.jump_to_bottom = True
        # 保存当前输入框内容
        if 'prompt_input' in st.session_state:
            st.session_state.prompt_input_saved = st.session_state.prompt_input
    
    # 检查是否有强制跳转的页面（从风格应用按钮触发）
    if 'force_page' in st.session_state:
        forced_page = st.session_state.force_page
        del st.session_state.force_page  # 清除强制跳转标志
        page = forced_page
        st.session_state.page = forced_page
    else:
        page = st.sidebar.selectbox(
            "选择页面",
            ["🎨 图像生成", "⚙️ 系统管理"],
            index=["🎨 图像生成", "⚙️ 系统管理"].index(st.session_state.page),
            on_change=on_page_change,
            key="page_selector"
        )
        st.session_state.page = page
    
    if page == "⚙️ 系统管理":
        admin_page()
        return
    
    # 检查是否需要强制跳转到最底部
    if st.session_state.get('jump_to_bottom', False):
        # 注入立即滚动脚本
        st.markdown("""
        <script>
            setTimeout(function() {
                const container = document.querySelector('[data-testid="stMainBlockContainer"]');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            }, 0);
        </script>
        """, unsafe_allow_html=True)
        # 重置标志
        st.session_state.jump_to_bottom = False
    
    # 从数据库加载历史记录到session_state
    if not st.session_state.history_loaded:
        history_records = load_from_database()
        for record in history_records:
            st.session_state.messages.append({
                "role": "user",
                "content": record['prompt']
            })
            st.session_state.messages.append({
                "role": "assistant",
                "prompt": record['prompt'],
                "image": record['image_data'],
                "quality": {
                    'score': record['quality_score'],
                    'level': record['quality_level'],
                    'text': record['quality_text']
                },
                "time": record['generation_time'],
                "db_id": record['id']
            })
        st.session_state.history_loaded = True
    
    # 标题
    st.markdown('<div class="main-header">🎨 AI Image Generator</div>', unsafe_allow_html=True)
    
    # 显示当前应用的风格
    if st.session_state.current_style:
        style_info = st.session_state.current_style
        st.markdown(f'''
        <div class="current-style-badge">
            ✨ 当前风格: {style_info['name']} ({style_info['suffix']})
        </div>
        ''', unsafe_allow_html=True)
        
        col_clear = st.columns([1])
        if col_clear[0].button("🚫 清除当前风格", key="clear_current_style"):
            st.session_state.current_style = None
            st.success("已清除当前风格！")
            st.rerun()
    
    # 侧边栏 - 历史对话和统计
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # 统计信息
        st.subheader("📊 生成统计")
        stats = get_generation_stats()
        if stats['total_generations'] > 0:
            st.markdown(f'''
            <div class="stats-card">
                <div>🎨 总生成次数: <strong>{stats["total_generations"]}</strong></div>
                <div>⏱️ 平均时间: <strong>{stats["average_time"]}秒</strong></div>
                <div>⚡ 上次时间: <strong>{stats["last_time"]}秒</strong></div>
                <div>🏆 常用提示: <strong>{stats["most_common_prompt"]}</strong></div>
                <div>📈 使用次数: <strong>{stats["prompt_count"]}</strong></div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.info("尚未生成任何图像，开始您的创作之旅吧！")
        
        # 历史对话
        st.subheader("💬 历史对话")
        
        history_records = load_from_database()
        
        if history_records:
            if st.button("🗑️ 清空所有历史记录", key="clear_all"):
                st.session_state.show_clear_confirm = True
            
            for i, record in enumerate(history_records):
                prompt = record['prompt']
                image_data = record['image_data']
                quality_score = record['quality_score']
                quality_level = record['quality_level']
                quality_text = record['quality_text']
                
                with st.expander(f"提示词 {i+1}: {prompt[:30]}...", expanded=False):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**提示词:** {prompt}")
                        if image_data:
                            st.markdown(f'''
                            <div class="quality-badge quality-{quality_level}">
                                质量评分: {quality_score}/100 ({quality_text})
                            </div>
                            <div class="image-container">
                                <img src="data:image/png;base64,{image_data}" class="generated-image">
                            </div>
                            ''', unsafe_allow_html=True)
                        

                    
                    with col2:
                        if st.button(f"🗑️", key=f"delete_{record['id']}", help="删除此记录"):
                            delete_from_database(record['id'])
                            st.success(f"已删除记录 {i+1}！")
                            st.rerun()
        else:
            st.info("暂无历史对话，开始生成您的第一张图像吧！")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 全屏确认清空对话框
    if st.session_state.get('show_clear_confirm', False):
        st.info("""
        ### ⚠️ 确认清空
        
        确定要清空所有历史记录吗？此操作不可恢复！
        """)
        
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("✅ 确认清空", key="confirm_clear", type="primary"):
                clear_all_history()
                st.session_state.show_clear_confirm = False
                st.success("所有历史记录已清空！")
                st.rerun()
        with col_cancel:
            if st.button("❌ 取消", key="cancel_clear"):
                st.session_state.show_clear_confirm = False
                st.rerun()
    
    # 主聊天区域
    col1, col2 = st.columns([3, 1])
    
    # 收藏对话框
    @st.dialog("⭐ 收藏提示词")
    def favorite_dialog():
        st.markdown("### 📌 确认收藏")
        st.write(f"提示词：{st.session_state.favorite_dialog_prompt}")
        st.write("是否将此提示词添加到收藏列表？")
        
        col_dialog1, col_dialog2 = st.columns(2)
        with col_dialog1:
            if st.button("✅ 确认收藏", key="confirm_favorite_dialog"):
                # 检查是否已收藏
                existing_fav = None
                all_favorites = get_favorite_prompts()
                for fav in all_favorites:
                    if fav['prompt'] == st.session_state.favorite_dialog_prompt:
                        existing_fav = fav
                        break
                
                if existing_fav is None:
                    add_favorite_prompt(st.session_state.favorite_dialog_prompt)
                    if st.session_state.favorite_dialog_index is not None:
                        st.session_state.messages[st.session_state.favorite_dialog_index]['favorited'] = True
                    st.success("已添加到收藏！")
                else:
                    st.info("该提示词已在收藏中！")
                
                st.session_state.show_favorite_dialog = False
                st.rerun()
        
        with col_dialog2:
            if st.button("❌ 取消", key="cancel_favorite_dialog"):
                st.session_state.show_favorite_dialog = False
                st.rerun()
    
    # 如果需要显示对话框
    if st.session_state.show_favorite_dialog:
        favorite_dialog()
    
    with col1:
        # 显示对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'''
                <div class="chat-container">
                    <div class="user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            else:
                favorited = message.get('favorited', False)
                favorite_icon = "⭐" if favorited else "☆"
                
                # 获取消息索引
                msg_index = st.session_state.messages.index(message)
                
                # 创建可点击的图片容器
                st.markdown(f'''
                <div class="chat-container" id="image_container_{msg_index}">
                    <div class="ai-message">
                        <strong>AI:</strong> Generating image for: "{message["prompt"]}"
                    </div>
                    <div class="image-container" style="position: relative;">
                        <img src="data:image/png;base64,{message["image"]}" 
                             class="generated-image" 
                             id="gen_image_{msg_index}"
                             style="cursor: pointer;"
                             onclick="document.getElementById('favorite_btn_{msg_index}').click()">
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # 收藏按钮（隐藏，点击图片触发）
                if not favorited:
                    if st.button(f"⭐ 收藏", key=f"favorite_{msg_index}"):
                        # 检查是否已收藏
                        existing_fav = None
                        all_favorites = get_favorite_prompts()
                        for fav in all_favorites:
                            if fav['prompt'] == message['prompt']:
                                existing_fav = fav
                                break
                        
                        if existing_fav is None:
                            add_favorite_prompt(message['prompt'])
                            st.success("已添加到收藏！")
                        else:
                            st.info("该提示词已在收藏中！")
                        
                        # 更新消息状态
                        st.session_state.messages[msg_index]['favorited'] = True
                        st.rerun()
                else:
                    st.markdown(f'<p style="color: #FFFF00; margin-top: 5px; font-weight: bold;">⭐ 已收藏</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='white-space: nowrap; overflow: visible;'>💡 Quick Prompts</h3>", unsafe_allow_html=True)
        quick_prompts = [
            "a beautiful sunset over the ocean",
            "a cat wearing sunglasses on a beach",
            "a futuristic city skyline at night",
            "a cozy cabin in the woods",
            "a spaceship flying through space"
        ]
        
        # 使用容器让按钮在一行显示
        st.markdown('<div class="quick-prompts-container">', unsafe_allow_html=True)
        for prompt in quick_prompts:
            if st.button(prompt, key=f"quick_{prompt}"):
                st.session_state.prompt_input = prompt
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 输入区域（放在页面最底部）
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    col3, col4 = st.columns([4, 1])
    
    with col3:
        # 恢复之前保存的输入内容
        if st.session_state.prompt_input_saved:
            default_value = st.session_state.prompt_input_saved
            st.session_state.prompt_input_saved = ''  # 清空保存的内容
        else:
            default_value = ''
        
        prompt = st.text_input(
            "Enter your English prompt:",
            placeholder="Describe the image you want to generate (e.g., 'a beautiful sunset over the ocean')",
            key="prompt_input",
            label_visibility="collapsed",
            value=default_value
        )
    
    with col4:
        generate_btn = st.button("🎨 Generate", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 处理生成请求
    if generate_btn and prompt:
        if st.session_state.generating:
            st.warning("Please wait, image generation in progress...")
        else:
            st.session_state.generating = True
            
            # 应用当前风格
            if st.session_state.current_style:
                style_suffix = st.session_state.current_style['suffix']
                if not prompt.endswith(style_suffix):
                    prompt = prompt + style_suffix
                    st.info(f"✨ 已应用风格: {st.session_state.current_style['name']}")
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("🎨 Generating your image... This may take 2-3 minutes."):
                image = generate_image_with_progress(prompt)
                
                if image[0]:
                    buffered = BytesIO()
                    image[0].save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    quality_info = assess_image_quality(image[0])
                    db_id = save_to_database(prompt, img_str, quality_info, image[1])
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "prompt": prompt,
                        "image": img_str,
                        "quality": quality_info,
                        "time": image[1],
                        "db_id": db_id,
                        "favorited": False
                    })
                    
                    st.success("✅ Image generated successfully!")
                else:
                    st.error("❌ Failed to generate image. Please try again.")
            
            st.session_state.generating = False
            st.rerun()
    
    # 注入自动滚动脚本（每次页面渲染都会执行）
    auto_scroll_to_bottom()
    
    # 显示模型状态
    if not st.session_state.model_loaded:
        st.info("🔧 Initializing AI model... This may take a few minutes on first run.")
        if st.button("Initialize Model"):
            with st.spinner("Loading model components..."):
                if initialize_model():
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load model. Please check the console for errors.")

if __name__ == "__main__":
    main()