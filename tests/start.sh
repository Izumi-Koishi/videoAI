#!/usr/bin/env bash
# ============================================================
# videoAI 一键启动脚本 (Linux / macOS)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  videoAI - 校园监控视频 AI 智能问答系统${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ----------------------------------------------------------
# 1. 检查 Python
# ----------------------------------------------------------
echo -e "${YELLOW}[1/4] 检查 Python 环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}  [错误] 未找到 Python3，请安装 Python 3.10+${NC}"
    echo "   Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "   macOS:         brew install python@3.11"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "  Python ${PYTHON_VERSION}"

# ----------------------------------------------------------
# 2. 检查虚拟环境
# ----------------------------------------------------------
if [ -f ".venv/bin/python" ]; then
    echo -e "  ${GREEN}[信息]${NC} 检测到 .venv 虚拟环境"
    read -p "  是否使用虚拟环境？[Y/n] " USE_VENV
    if [ "$USE_VENV" != "n" ] && [ "$USE_VENV" != "N" ]; then
        source .venv/bin/activate
        echo "  虚拟环境已激活"
    fi
fi

# ----------------------------------------------------------
# 3. 安装依赖
# ----------------------------------------------------------
echo ""
echo -e "${YELLOW}[2/4] 检查并安装依赖...${NC}"
read -p "  是否安装/更新依赖？[Y/n] " INSTALL_DEPS
if [ "$INSTALL_DEPS" != "n" ] && [ "$INSTALL_DEPS" != "N" ]; then
    echo "  正在安装依赖..."
    pip install -r tests/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || \
    pip install -r tests/requirements.txt
    echo -e "  ${GREEN}[完成]${NC} 依赖安装完成"
else
    echo "  跳过依赖安装"
fi

# ----------------------------------------------------------
# 4. 检查数据文件
# ----------------------------------------------------------
echo ""
echo -e "${YELLOW}[3/4] 检查数据文件...${NC}"
DATA_OK=true
for file in "data/data/campus_knowledge_base.json" "data/data/campus_test_qa.json"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}[√]${NC} $file"
    else
        echo -e "  ${RED}[×]${NC} 缺少文件: $file"
        DATA_OK=false
    fi
done

# ----------------------------------------------------------
# 5. 环境验证
# ----------------------------------------------------------
echo ""
echo -e "${YELLOW}[4/4] 快速环境验证...${NC}"
python3 tests/main.py --mode verify || {
    echo ""
    echo -e "${YELLOW}  [警告] 环境验证未完全通过，部分功能可能不可用${NC}"
}

# ----------------------------------------------------------
# 6. 选择运行模式
# ----------------------------------------------------------
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  请选择运行模式:${NC}"
echo -e "${BLUE}============================================================${NC}"
echo "  [1] 验证环境 (推荐首先运行)"
echo "  [2] 处理测试视频 (YOLO 检测 + 向量入库)"
echo "  [3] 运行集成测试"
echo "  [4] 完整管线 (视频处理 + 自动化测试)"
echo "  [5] 交互式命令行问答"
echo "  [6] 启动 Web 界面 (Gradio)"
echo "  [7] 退出"
echo -e "${BLUE}============================================================${NC}"
read -p "请输入选项 [1-7]: " MODE

case $MODE in
    1)
        python3 tests/main.py --mode verify
        ;;
    2)
        python3 tests/main.py --mode process
        ;;
    3)
        read -p "跳过 LLM 测试？[Y/n] " SKIP
        if [ "$SKIP" = "n" ] || [ "$SKIP" = "N" ]; then
            python3 tests/integration_test.py
        else
            python3 tests/integration_test.py --skip-llm
        fi
        echo ""
        echo -e "${GREEN}测试报告已生成: tests/output/integration_test_report.md${NC}"
        ;;
    4)
        python3 tests/main.py --mode pipeline
        ;;
    5)
        python3 tests/main.py --mode interactive
        ;;
    6)
        echo ""
        echo -e "${GREEN}启动 Web 界面...${NC}"
        echo "  本地地址: http://127.0.0.1:7860"
        echo "  按 Ctrl+C 停止服务"
        echo ""
        cd llm_gradio
        python3 app.py
        ;;
    7)
        echo "退出"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        ;;
esac

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  videoAI 运行结束${NC}"
echo -e "${BLUE}============================================================${NC}"
