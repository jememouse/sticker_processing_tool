<template>
  <div class="app">
    <!-- 头部 -->
    <header class="header">
      <div class="logo">
        <span class="logo-icon">✨</span>
        <h1>智能贴纸生成器</h1>
      </div>
      <p class="tagline">AI 驱动 · 印刷级输出</p>
    </header>
    
    <!-- 主容器 -->
    <main class="main-container">
      <!-- 左侧面板 -->
      <aside class="sidebar">
        <!-- 上传区域 (仅在未上传时显示) -->
        <FileUpload 
          v-if="!uploadedFile" 
          @file-selected="handleFileSelected"
        />
        
        <!-- 已上传文件信息 -->
        <div v-else class="uploaded-file">
          <div class="file-info">
            <span class="file-icon">📁</span>
            <div class="file-details">
              <span class="file-name">{{ uploadedFile.filename }}</span>
              <button class="change-file" @click="resetUpload">更换图片</button>
            </div>
          </div>
        </div>
        
        <!-- 参数面板 -->
        <ParamPanel 
          :can-generate="!!uploadedFile"
          :loading="isGenerating"
          @generate="handleGenerate"
        />
        
        <!-- 下载区域 -->
        <DownloadBar 
          :task-id="currentTaskId"
          :task-completed="taskStatus === 'completed'"
        />
      </aside>
      
      <!-- 右侧预览区域 -->
      <section class="canvas-area">
        <PreviewArea 
          :image-url="previewUrl"
          :loading="isGenerating"
          :progress="taskProgress"
          :status="taskStatus"
        />
      </section>
    </main>
    
    <!-- 错误提示 -->
    <div v-if="errorMessage" class="error-toast">
      <span>{{ errorMessage }}</span>
      <button @click="errorMessage = ''">✕</button>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import FileUpload from './components/FileUpload.vue'
import ParamPanel from './components/ParamPanel.vue'
import PreviewArea from './components/PreviewArea.vue'
import DownloadBar from './components/DownloadBar.vue'
import { 
  uploadImage, 
  generateSticker, 
  getTaskStatus,
  getOriginalPreviewUrl,
  getDownloadUrl
} from './api/sticker'

// 状态
const uploadedFile = ref(null)
const currentTaskId = ref('')
const taskStatus = ref('')
const taskProgress = ref(0)
const isGenerating = ref(false)
const errorMessage = ref('')

// 预览 URL
const previewUrl = computed(() => {
  if (taskStatus.value === 'completed' && currentTaskId.value) {
    return getDownloadUrl(currentTaskId.value, 'preview')
  }
  if (uploadedFile.value) {
    return getOriginalPreviewUrl(uploadedFile.value.file_id)
  }
  return ''
})

// 处理文件选择
async function handleFileSelected(file) {
  try {
    errorMessage.value = ''
    const result = await uploadImage(file)
    uploadedFile.value = result
    // 重置之前的任务状态
    currentTaskId.value = ''
    taskStatus.value = ''
    taskProgress.value = 0
  } catch (err) {
    errorMessage.value = err.response?.data?.detail || '上传失败，请重试'
  }
}

// 重置上传
function resetUpload() {
  uploadedFile.value = null
  currentTaskId.value = ''
  taskStatus.value = ''
  taskProgress.value = 0
}

// 处理生成
async function handleGenerate(params) {
  if (!uploadedFile.value) return
  
  try {
    errorMessage.value = ''
    isGenerating.value = true
    taskProgress.value = 0
    taskStatus.value = 'pending'
    
    // 调用生成 API
    const result = await generateSticker({
      file_id: uploadedFile.value.file_id,
      ...params
    })
    
    currentTaskId.value = result.task_id
    
    // 轮询任务状态
    pollTaskStatus()
  } catch (err) {
    errorMessage.value = err.response?.data?.detail || '生成失败，请重试'
    isGenerating.value = false
  }
}

// 轮询任务状态
async function pollTaskStatus() {
  if (!currentTaskId.value) return
  
  try {
    const status = await getTaskStatus(currentTaskId.value)
    taskStatus.value = status.status
    taskProgress.value = status.progress
    
    if (status.status === 'completed') {
      isGenerating.value = false
    } else if (status.status === 'failed') {
      isGenerating.value = false
      errorMessage.value = status.error || '处理失败'
    } else {
      // 继续轮询
      setTimeout(pollTaskStatus, 500)
    }
  } catch (err) {
    isGenerating.value = false
    errorMessage.value = '获取状态失败'
  }
}
</script>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  text-align: center;
  padding: 32px 24px;
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.1) 0%, transparent 100%);
}

.logo {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 8px;
}

.logo-icon {
  font-size: 2rem;
}

.logo h1 {
  font-size: 1.75rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.tagline {
  color: var(--color-text-secondary);
  font-size: 0.875rem;
}

.main-container {
  flex: 1;
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 24px;
  padding: 0 24px 24px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.sidebar {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.uploaded-file {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 16px;
  padding: 20px;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.file-icon {
  font-size: 2rem;
}

.file-details {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.file-name {
  font-weight: 500;
  color: var(--color-text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.change-file {
  background: none;
  border: none;
  color: var(--color-primary);
  font-size: 0.875rem;
  cursor: pointer;
  padding: 0;
  text-align: left;
}

.change-file:hover {
  text-decoration: underline;
}

.canvas-area {
  display: flex;
  flex-direction: column;
}

.error-toast {
  position: fixed;
  bottom: 24px;
  left: 50%;
  transform: translateX(-50%);
  background: #ef4444;
  color: white;
  padding: 16px 24px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 8px 32px rgba(239, 68, 68, 0.4);
  z-index: 1000;
}

.error-toast button {
  background: none;
  border: none;
  color: white;
  cursor: pointer;
  font-size: 1rem;
}

/* 响应式 */
@media (max-width: 900px) {
  .main-container {
    grid-template-columns: 1fr;
  }
  
  .canvas-area {
    order: -1;
    min-height: 300px;
  }
}
</style>
