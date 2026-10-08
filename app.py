from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests, re, json, os, sys

app = Flask(__name__)
CORS(app)

# ============================================================
# 配置区（密钥、API 地址、模型都在这里，按需修改）
# ============================================================

def _load_dotenv(path=None):
    """极简 .env 读取器：把 .env 里的 KEY=VALUE 读进环境变量（不引入第三方依赖）。

    推荐做法：在项目目录 D:\\project\\Halsey Gao 下创建 .env 文件，内容示例：
        VERCEL_API_KEY=vck_你的真实Key
    .env 不要提交到代码仓库；app.py 顶部的占位符只是在找不到 .env 时兜底。
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


_load_dotenv()

# ChatAnywhere（OpenAI 兼容接口，主力视觉模型平台）
API_URL = "https://api.chatanywhere.tech/v1/chat/completions"
# 密钥优先读环境变量 CHATANYWHERE_API_KEY（来自 .env 文件）；
# 读不到时使用下方占位符——也可以直接把占位符换成真实值
API_KEY = os.environ.get("CHATANYWHERE_API_KEY", "你的ChatAnywhere付费Key")

# 模型阶梯：第一层用主力模型，自我评估不准后升级到第三层的更强模型
# 注意：ChatAnywhere 的模型名用横杠连接版本号（如 claude-sonnet-5-5，不是 5.5）
MODEL_CHEAP = "claude-sonnet-5-5"  # 第一层：主力模型（付费）
MODEL_STRONG = "claude-opus-5-5"   # 第三层：更强兜底模型（可自行替换）
# 日志里显示用的友好名称（方便评委阅读）
MODEL_DISPLAY = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5（主力模型）",
    "claude-opus-5-5": "Claude Opus 5.5（高级模型）",
}

# ---------------- 提示词 ----------------

AGENT_BASE = "你是 AdClear Agent，一个帮助老年人和视障用户操作手机的智能助手。"

DETECT_PROMPT = AGENT_BASE + """
请观察这张手机屏幕截图（分辨率 {SIZE} 像素），完成任务：
1. 判断当前屏幕是否存在开屏广告（全屏或大面积推广画面）
2. 如果存在，找到关闭/跳过按钮（如 ×、关闭、跳过、跳过广告，可能带倒计时）

只输出 JSON，不要输出其他任何内容：
{"has_ad": true或false, "bbox": [x1,y1,x2,y2]或null, "label": "按钮上的文字或符号", "confidence": 0到1之间的小数, "reason": "一句话说明依据"}

要求：
- bbox 坐标以图片左上角为原点，单位像素（x 在 0 到图宽、y 在 0 到图高之间），必须完整框住按钮
- 找不到按钮时 bbox 为 null 且 confidence 低于 0.5
- 不确定时如实给出较低的 confidence，不要猜测
"""

RETRY_PROMPT = AGENT_BASE + """
请再次仔细观察这张手机屏幕截图（分辨率 {SIZE} 像素）。上一次分析未能可靠地找到关闭开屏广告的按钮。

请特别留意：
- 屏幕四个角落，尤其是右上角和左上角
- 可能是"跳过"、"跳过广告"、"关闭"文字的按钮（有时带倒计时，如"跳过 5"）
- 颜色与背景接近、半透明、尺寸很小的按钮
- 圆形或方形的小图标

只输出 JSON，不要输出其他任何内容：
{"has_ad": true或false, "bbox": [x1,y1,x2,y2]或null, "label": "按钮上的文字或符号", "confidence": 0到1之间的小数, "reason": "一句话说明依据"}

要求：bbox 坐标以图片左上角为原点，单位像素（x 在 0 到图宽、y 在 0 到图高之间）。
"""

VERIFY_PROMPT = AGENT_BASE + """
请观察这张手机屏幕截图，只回答一个问题：当前屏幕是否还存在开屏广告（全屏或大面积推广画面）？

只输出 JSON，不要输出其他任何内容：
{"has_ad": true或false, "bbox": null, "label": "", "confidence": 0到1之间的小数, "reason": "一句话说明"}
"""

LOCALIZE_PROMPT = AGENT_BASE + """
这是手机屏幕的局部放大截图，已经确定附近存在开屏广告的关闭/跳过按钮。请在这张局部图中精确定位这个按钮（如 ×、关闭、跳过、跳过广告，可能带倒计时）。

只输出 JSON，不要输出其他任何内容：
{"has_ad": true, "bbox": [x1,y1,x2,y2], "label": "按钮上的文字或符号", "confidence": 0到1之间的小数, "reason": "一句话说明依据"}

要求：
- bbox 坐标以这张局部图左上角为原点，单位像素，必须完整框住按钮
- 局部图中确实找不到按钮时，如实输出 bbox 为 null 且 confidence 低于 0.5
"""

# ---------------- 工具函数 ----------------

def _key_ready():
    """占位符 Key（含"你的"字样）视为未配置。"""
    return "你的" not in API_KEY


def _model_display(model):
    """模型 ID → 日志里的友好名称。"""
    return MODEL_DISPLAY.get(model, model)


def _extract_balanced(text):
    """从文本中提取第一个『括号配平』的 JSON 对象（跳过字符串内的括号）。"""
    start = text.find('{')
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        start = text.find('{', start + 1)
    return None


def _escape_newlines_in_strings(s):
    """把引号字符串内部的裸换行转义为 \\n（模型偶尔在 reason 里直接换行）。"""
    out = []
    in_str = False
    escape = False
    for ch in s:
        if in_str:
            if escape:
                out.append(ch)
                escape = False
            elif ch == '\\':
                out.append(ch)
                escape = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch in '\r\n':
                out.append('\\n')
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return ''.join(out)


def _normalize_punct(s):
    """全角标点归一化（模型用中文标点输出 JSON 时救回来）。"""
    table = {'：': ':', '，': ',', '；': ';', '（': '(', '）': ')', '【': '[', '】': ']'}
    for k, v in table.items():
        s = s.replace(k, v)
    return s


def _try_loads(s):
    """宽松解析字符串为 dict：把各修复手段按任意组合叠加后逐一尝试。"""
    transforms = (
        _normalize_punct,                                              # 全角标点归一化
        _escape_newlines_in_strings,                                   # 字符串内裸换行转义
        lambda x: re.sub(r',\s*([}\]])', r'\1', x),                    # 去尾逗号
        lambda x: x.replace('“', '"').replace('”', '"')                # 中文引号
                   .replace('‘', "'").replace('’', "'"),
    )
    current = {s}
    for t in transforms:
        for v in list(current):
            current.add(t(v))
    for v in current:
        try:
            obj = json.loads(v)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _extract_json(text):
    """从模型输出中尽量鲁棒地提取第一个 JSON 对象，失败返回 None。

    容错场景：markdown 代码块围栏（```json ... ```）、JSON 前后有解释性文字、
    尾逗号、中文引号、字符串内含有 } 等。
    """
    if not text:
        return None
    candidates = []
    # 1) 整段就是 JSON
    candidates.append(text.strip())
    # 2) 去掉 markdown 围栏后的整段
    candidates.append(re.sub(r'```[a-zA-Z]*\s*', '', text).strip())
    # 3) 括号配平提取（JSON 前后混有文字时）
    balanced = _extract_balanced(text)
    if balanced:
        candidates.append(balanced)
    # 4) 兜底：贪婪正则（第一个 { 到最后一个 }）
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        candidates.append(m.group())
    for c in candidates:
        result = _try_loads(c)
        if result is not None:
            return result
    return None


def _call_vision(image_url, prompt, model):
    """调用视觉模型（当前为智谱开放平台，OpenAI 兼容格式），返回 (ok, content, error, auth_error)。"""
    if not _key_ready():
        return False, None, "API Key 未配置：请在项目目录的 .env 文件中填入真实的 CHATANYWHERE_API_KEY（或修改 app.py 顶部的占位符）", True
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt}
            ]
        }],
        # 要求模型输出纯 JSON（已实测 ChatAnywhere 支持；即使模型仍加围栏，解析器也能容错）
        "response_format": {"type": "json_object"}
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        r = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    except requests.exceptions.RequestException as e:
        return False, None, f"网络请求失败：{e}", False
    if r.status_code == 401:
        return False, None, "API Key 无效或无权限，请检查 .env 中的 CHATANYWHERE_API_KEY（可到 chatanywhere.tech 控制台查看）", True
    if r.status_code == 404:
        return False, None, (f"模型 {model} 不存在（HTTP 404）：请确认该模型名可用，"
                             f"必要时修改 app.py 顶部的 MODEL_CHEAP / MODEL_STRONG。详情：{r.text[:150]}"), False
    if r.status_code != 200:
        return False, None, f"模型接口返回 {r.status_code}：{r.text[:200]}", False
    try:
        content = r.json()['choices'][0]['message']['content']
    except (KeyError, IndexError, ValueError):
        return False, None, "模型响应格式异常", False
    return True, content, None, False


def _normalize(parsed, width=None, height=None):
    """把模型输出的 JSON 规整成接口统一格式。

    视觉模型有时会返回 0~1 之间的归一化坐标，这里统一换算成像素坐标。
    """
    bbox = parsed.get('bbox')
    if not (isinstance(bbox, list) and len(bbox) == 4
            and all(isinstance(v, (int, float)) for v in bbox)):
        bbox = None
    if bbox and width and height:
        x1, y1, x2, y2 = bbox
        # 全部在 0~1 之间且不是退化框 → 视为归一化坐标，换算成像素
        if all(0 <= v <= 1 for v in bbox) and x2 > x1 and y2 > y1:
            bbox = [x1 * width, y1 * height, x2 * width, y2 * height]
    try:
        confidence = float(parsed.get('confidence', 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "ok": True,
        "has_ad": bool(parsed.get('has_ad', False)),
        "bbox": bbox,
        "label": str(parsed.get('label', '')),
        "confidence": confidence,
        "reason": str(parsed.get('reason', ''))
    }


def is_result_accurate(bbox, raw_text, confidence, width=720, height=1560):
    """自我评估：判断识别结果是否可靠。返回 (是否可靠, 原因列表)。

    判定"不可靠"的规则：
    1. bbox 为 None
    2. bbox 的 4 个坐标值越界或顺序错误
    3. bbox 宽或高小于 10 像素
    4. 模型输出文字里没有「跳过/关闭/X」等关键词
    5. 有 confidence 且低于 0.6
    """
    reasons = []
    text = raw_text or ""
    if bbox is None:
        reasons.append("bbox 为空（未定位到关闭按钮）")
    else:
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            reasons.append("bbox 坐标越界或顺序错误")
        elif (x2 - x1) < 10 or (y2 - y1) < 10:
            reasons.append("bbox 宽或高小于 10 像素")
    if not any(kw in text for kw in ("跳过", "关闭", "X", "×")):
        reasons.append("输出文字缺少「跳过/关闭/X」等关键词")
    if confidence is not None and confidence < 0.6:
        reasons.append(f"置信度 {confidence:.2f} 低于 0.6")
    return len(reasons) == 0, reasons


def _escalate(img, prompt, width, height, logs):
    """第三层：升级兜底——用高级模型重新识别一次。"""
    logs.append(f"[Agent] 正在切换至高级模型 ({_model_display(MODEL_STRONG)}) 兜底...")
    ok, content, error, auth_error = _call_vision(img, prompt, MODEL_STRONG)
    if not ok:
        return {"ok": False, "auth_error": auth_error, "error": error,
                "used_model": MODEL_STRONG,
                "agent_logs": logs + [f"[错误] {error}"]}
    parsed = _extract_json(content)
    if parsed is None:
        return {"ok": False, "auth_error": False, "error": "高级模型输出无法解析为 JSON",
                "used_model": MODEL_STRONG, "agent_logs": logs, "raw": content[:300]}
    resp = _normalize(parsed, width, height)
    resp["used_model"] = MODEL_STRONG
    resp["agent_logs"] = logs + ["[结果] 识别成功！高亮展示中。"]
    return resp


def _perceive_with_ladder(img, prompt, width, height, first_log=None):
    """分层决策核心：快速模型 → 自我评估 → 不准则自动升级高级模型。

    解析失败时先用重试提示词让快速模型再试一次（格式抖动很常见），
    重试仍失败才升级到高级模型——避免频繁触发昂贵升级，省时省钱。
    """
    w, h = int(width or 720), int(height or 1560)
    if first_log is None:
        first_log = f"[Agent] 正在调用快速模型 ({_model_display(MODEL_CHEAP)}) 进行识别..."
    logs = [first_log]

    ok, content, error, auth_error = _call_vision(img, prompt, MODEL_CHEAP)
    if not ok:
        return {"ok": False, "auth_error": auth_error, "error": error,
                "used_model": MODEL_CHEAP,
                "agent_logs": logs + [f"[错误] {error}"]}

    parsed = _extract_json(content)
    if parsed is None:
        # 第一层容错：格式抖动 → 快速模型用重试提示词再试一次
        logs.append(f"[结果] 快速模型输出无法解析为 JSON（原始输出：{content[:120]}）")
        logs.append("[Agent 决策] 输出格式异常，先用重试提示词再试一次快速模型……")
        retry_prompt = RETRY_PROMPT.replace("{SIZE}", f"{w}×{h}")
        ok2, content2, error2, auth2 = _call_vision(img, retry_prompt, MODEL_CHEAP)
        if ok2:
            parsed = _extract_json(content2)
            if parsed is not None:
                logs.append("[结果] 快速模型重试成功，输出已解析。")
            else:
                logs.append(f"[结果] 快速模型重试仍无法解析（原始输出：{content2[:120]}）")
                logs.append("[Agent 决策] 快速模型结果不理想，触发升级策略。")
                return _escalate(img, prompt, w, h, logs)
        else:
            logs.append(f"[错误] 快速模型重试调用失败：{error2}")
            logs.append("[Agent 决策] 触发升级策略。")
            return _escalate(img, prompt, w, h, logs)

    resp = _normalize(parsed, w, h)
    if not resp["has_ad"]:
        resp["used_model"] = MODEL_CHEAP
        resp["agent_logs"] = logs + ["[结果] 当前屏幕没有检测到开屏广告。"]
        return resp

    # 第二层：自我反思与评估
    accurate, reasons = is_result_accurate(resp["bbox"], content, resp["confidence"], w, h)
    if not accurate:
        logs.append("[结果] " + "；".join(reasons) + "。")
        logs.append("[Agent 决策] 快速模型结果不理想，触发升级策略。")
        return _escalate(img, prompt, w, h, logs)

    resp["used_model"] = MODEL_CHEAP
    resp["agent_logs"] = logs + ["[结果] 快速模型识别成功，无需升级。"]
    return resp

# ---------------- 路由 ----------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """原始截图识别接口（开发者测试用，保留原逻辑，走 Vercel AI Gateway）。"""
    data = request.json
    img_b64 = data['image']
    prompt = (
        '请找出图中开屏广告的关闭/跳过按钮（通常是右上角的X、跳过文字或倒计时按钮），'
        '输出JSON：{"bbox":[x1,y1,x2,y2], "label":"跳过", "confidence":0.98}。'
        '坐标以图片左上角为原点，单位像素。'
        '如果找不到，输出 {"bbox":null}。只输出JSON，不要解释。'
    )
    payload = {
        "model": MODEL_CHEAP,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        content = r.json()['choices'][0]['message']['content']
        match = re.search(r'\{.*\}', content, re.S)
        if match:
            return jsonify(json.loads(match.group()))
        return jsonify({"bbox": None, "raw": content})
    except Exception as e:
        return jsonify({"bbox": None, "error": str(e)})


@app.route('/agent/perceive', methods=['POST'])
def agent_perceive():
    """Agent 的视觉感知接口（分层决策版）。

    mode:
      detect   —— 分层决策：快速模型 → 自我评估 → 不准自动升级高级模型
      retry    —— 同样分层决策，使用带提示的 retry 提示词
      escalate —— 跳过快速模型，直接使用高级模型
      verify   —— 快速模型确认广告是否还在（布尔判断，无需升级）
      localize —— 局部图精确定位（分层决策）

    返回字段包含 used_model（本次实际使用的模型）与 agent_logs（思考日志数组）。
    """
    data = request.json or {}
    img = data.get('image', '')
    mode = data.get('mode', 'detect')
    width = data.get('width')
    height = data.get('height')
    if not img:
        return jsonify({"ok": False, "auth_error": False, "error": "缺少截图", "agent_logs": []})

    size_text = f"{int(width or 720)}×{int(height or 1560)}"

    if mode == 'verify':
        # 验证也走分层决策：快速模型解析失败或判断存疑时自动升级，避免前端卡死在"验证未完成"
        prompt = VERIFY_PROMPT.replace("{SIZE}", size_text)
        first_log = f"[Agent] 正在调用快速模型 ({_model_display(MODEL_CHEAP)}) 验证屏幕状态..."
        return jsonify(_perceive_with_ladder(img, prompt, width, height, first_log=first_log))

    if mode == 'escalate':
        prompt = RETRY_PROMPT.replace("{SIZE}", size_text)
        logs = [f"[Agent] 正在调用高级模型 ({_model_display(MODEL_STRONG)}) 直接识别..."]
        ok, content, error, auth_error = _call_vision(img, prompt, MODEL_STRONG)
        if not ok:
            return jsonify({"ok": False, "auth_error": auth_error, "error": error,
                            "used_model": MODEL_STRONG,
                            "agent_logs": logs + [f"[错误] {error}"]})
        parsed = _extract_json(content)
        if parsed is None:
            return jsonify({"ok": False, "auth_error": False, "error": "高级模型输出无法解析为 JSON",
                            "used_model": MODEL_STRONG, "agent_logs": logs, "raw": content[:300]})
        resp = _normalize(parsed, width, height)
        resp["used_model"] = MODEL_STRONG
        resp["agent_logs"] = logs + ["[结果] 识别完成。"]
        return jsonify(resp)

    # detect / retry / localize：分层决策
    template = {"detect": DETECT_PROMPT, "retry": RETRY_PROMPT, "localize": LOCALIZE_PROMPT}.get(mode, DETECT_PROMPT)
    prompt = template.replace("{SIZE}", size_text)
    return jsonify(_perceive_with_ladder(img, prompt, width, height))


if __name__ == '__main__':
    # 本地开发：python app.py（debug 自动热重载 + 监视 .env）
    # 本地/演示关闭 debug：python app.py --no-debug
    # 云端（腾讯云 SCF Web 函数）：设环境变量 PORT=9000、FLASK_DEBUG=0，启动命令用 python app.py
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "") != "0" and '--no-debug' not in sys.argv
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True,
            extra_files=['.env'] if debug else None)
