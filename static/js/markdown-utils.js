class MarkdownRenderer {
    constructor() {
        this.initializeMarked();
    }

    initializeMarked() {
        if (typeof marked !== 'undefined') {
            // 配置 marked
            marked.setOptions({
                highlight: function(code, language) {
                    if (typeof hljs !== 'undefined' && language && hljs.getLanguage(language)) {
                        try {
                            return hljs.highlight(code, { language }).value;
                        } catch (__) {}
                    }
                    if (typeof hljs !== 'undefined') {
                        return hljs.highlightAuto(code).value;
                    }
                    return code;
                },
                breaks: true,
                gfm: true,
                sanitize: false
            });
        }
    }

    render(markdownText) {
        if (!markdownText || typeof marked === 'undefined') {
            return this.escapeHtml(markdownText || '');
        }
        
        try {
            return marked.parse(markdownText);
        } catch (error) {
            console.error('Markdown 渲染错误:', error);
            return this.escapeHtml(markdownText);
        }
    }

    escapeHtml(text) {
        if (text == null || text === undefined || text === '') {
            return '';
        }
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
    renderResultContent(result) {
        // 检查传入的是对象还是字符串
        let content = '';
        
        if (typeof result === 'string') {
            content = result;
        } else if (typeof result === 'object' && result !== null) {
            // 从对象中构建内容
            content = this.buildContentFromResult(result);
        } else {
            content = '无内容';
        }
        
        // 转义HTML
        const escapedContent = this.escapeHtml(content);
        
        // 使用marked解析markdown
        const html = marked.parse(escapedContent);
        
        // 返回HTML
        return html;
    }
    buildContentFromResult(result) {
        let content = '';
        
        // 添加视觉分析结果
        if (result.vision_analysis) {
            content += `**场景描述**: ${result.vision_analysis}\n\n`;
        }
        
        const isAlarm = result.is_alarm === '是' || (result.alarm_level && result.alarm_level !== '无');

        // 添加报警信息
        if (isAlarm) {
            content += `**🚨 报警状态**: ${result.alarm_level || '未知'}级报警\n\n`;
        } else {
            content += `**✅ 报警状态**: 无报警\n\n`;
        }
        
        // 仅报警时显示原因，无报警时完全隐藏该行
        if (isAlarm && result.alarm_reason) {
            content += `**📋 报警原因**: ${result.alarm_reason}\n\n`;
        }
        
        // 添加风险评估
        if (result.risk_assessment) {
            content += `**⚠️ 风险评估**: ${result.risk_assessment}\n\n`;
        }
        
        // 添加建议
        if (result.recommendation) {
            content += `**💡 处置建议**: ${result.recommendation}\n\n`;
        }
        
        // 添加置信度
        if (result.confidence !== undefined) {
            content += `**📊 置信度**: ${(result.confidence * 100).toFixed(1)}%\n\n`;
        }

        // 添加时间戳
        if (result.timestamp) {
            content += `*${result.timestamp}*`;
        }
        
        return content;
    }
    createMarkdownContainer(content) {
        const toolbar = this.createToolbarHTML(content);
        const renderedContent = this.render(content);
        
        return `
            ${toolbar}
            <div class="markdown-content">${renderedContent}</div>
        `;
    }

    createToolbarHTML(content) {
        return `
            <div class="markdown-toolbar">
                <button onclick="markdownRenderer.copyToClipboard('${this.escapeForAttribute(content)}')">📋 复制</button>
                <button onclick="markdownRenderer.toggleFullscreen(this.parentNode.parentNode)">🔍 全屏</button>
            </div>
        `;
    }

    escapeForAttribute(text) {
        return text.replace(/'/g, '&#39;').replace(/"/g, '&quot;').replace(/\n/g, '\\n');
    }

    isMarkdownContent(text) {
        if (!text) return false;
        
        // 简单的 markdown 格式检测
        const markdownPatterns = [
            /^#+\s/m,           // 标题
            /\*\*.*?\*\*/,      // 粗体
            /\*.*?\*/,          // 斜体
            /```[\s\S]*?```/,   // 代码块
            /`.*?`/,            // 行内代码
            /^\s*[-*+]\s/m,     // 无序列表
            /^\s*\d+\.\s/m,     // 有序列表
            /^\s*>\s/m,         // 引用
            /\[.*?\]\(.*?\)/    // 链接
        ];
        
        return markdownPatterns.some(pattern => pattern.test(text));
    }

    copyToClipboard(text) {
        // 解码 HTML 实体
        const decodedText = text.replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/\\n/g, '\n');
        
        if (navigator.clipboard) {
            navigator.clipboard.writeText(decodedText).then(() => {
                this.showToast('内容已复制到剪贴板');
            }).catch(() => {
                this.fallbackCopyTextToClipboard(decodedText);
            });
        } else {
            this.fallbackCopyTextToClipboard(decodedText);
        }
    }

    fallbackCopyTextToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.top = '0';
        textArea.style.left = '0';
        textArea.style.position = 'fixed';
        
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            this.showToast('内容已复制到剪贴板');
        } catch (err) {
            console.error('复制失败:', err);
            this.showToast('复制失败，请手动复制');
        }
        
        document.body.removeChild(textArea);
    }

    toggleFullscreen(element) {
        if (!element) return;
        
        element.classList.toggle('fullscreen-modal');
        
        if (element.classList.contains('fullscreen-modal')) {
            // 全屏模式
            const toolbar = element.querySelector('.markdown-toolbar button:last-child');
            if (toolbar) toolbar.textContent = '❌ 退出全屏';
        } else {
            // 退出全屏
            const toolbar = element.querySelector('.markdown-toolbar button:last-child');
            if (toolbar) toolbar.textContent = '🔍 全屏';
        }
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 12px 24px;
            border-radius: 6px;
            z-index: 10000;
            font-size: 14px;
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            if (document.body.contains(toast)) {
                document.body.removeChild(toast);
            }
        }, 2000);
    }
}

// 创建全局实例
const markdownRenderer = new MarkdownRenderer();
