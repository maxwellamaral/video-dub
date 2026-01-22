<script setup>
import { ref, onMounted, nextTick, watch } from 'vue'
import axios from 'axios'

const file = ref(null)
const motor = ref('mms')
const encoding = ref('rapido')
const logs = ref([])
const status = ref('idle') // idle, uploading, downloading, processing, done, error
const progress = ref(0)
const videoUrl = ref(null)
const logContainer = ref(null)
const ws = ref(null)

// YouTube support
const inputMode = ref('upload') // 'upload' or 'youtube'
const youtubeUrl = ref('')
const isValidYoutubeUrl = ref(false)

// Stepper Logic
const currentStep = ref(0)
const steps = [
  { id: 1, label: 'Extração', icon: '🎵' },
  { id: 2, label: 'Transcrição', icon: '📝' },
  { id: 3, label: 'Tradução', icon: '🌐' },
  { id: 4, label: 'Síntese TTS', icon: '🗣️' },
  { id: 5, label: 'Edição', icon: '🎬' }
]

const BACKEND_URL = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws'

onMounted(() => {
  connectWebSocket()
})

// Auto-scroll quando logs mudam
watch(logs, () => {
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
}, { deep: true })

const connectWebSocket = () => {
  try {
    ws.value = new WebSocket(WS_URL)
    ws.value.onopen = () => processLog("🔌 Conectado ao servidor de logs.")
    ws.value.onmessage = (event) => {
      processLog(event.data)
    }
    ws.value.onclose = () => {
      processLog("⚠️ Conexão perdida. Reconectando...")
      setTimeout(connectWebSocket, 3000)
    }
  } catch (e) {
    console.error("WS Error", e)
  }
}

const processLog = (msg) => {
  // Parsing de Progresso
  if (msg.startsWith("PROGRESS:")) {
    const val = parseInt(msg.split(":")[1].trim())
    progress.value = val
    return // Não adiciona ao log de texto
  }

  // Evitar duplicatas exatas consecutivas se desejar
  if (logs.value.length > 0 && logs.value[logs.value.length - 1] === msg) return;

  logs.value.push(msg)

  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })

  // Detect Step
  if (msg.includes("1. Extraindo")) { currentStep.value = 1; progress.value = 0 }
  else if (msg.includes("2. Transcrevendo")) currentStep.value = 2
  else if (msg.includes("3. Traduzindo")) currentStep.value = 3
  else if (msg.includes("4. Sintetizando")) currentStep.value = 4
  else if (msg.includes("5. Editando")) currentStep.value = 5
  else if (msg.includes("✅ Pipeline concluída")) { currentStep.value = 6; progress.value = 100 }
}

const handleFileChange = (e) => {
  if (e.target.files && e.target.files[0]) {
    file.value = e.target.files[0]
  }
}

const validateYoutubeUrl = () => {
  const youtubeRegex = /(https?:\/\/)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)\/(watch\?v=|embed\/|v\/|.+\?v=)?([^&=%\?]{11})/
  isValidYoutubeUrl.value = youtubeRegex.test(youtubeUrl.value)
}

const downloadFromYoutube = async () => {
  if (!isValidYoutubeUrl.value) {
    alert('Por favor, insira uma URL válida do YouTube!')
    return
  }

  status.value = 'downloading'
  logs.value = []
  currentStep.value = 0
  videoUrl.value = null
  processLog('🎥 Iniciando download do YouTube...')

  const formData = new FormData()
  formData.append('url', youtubeUrl.value)

  try {
    const res = await axios.post(`${BACKEND_URL}/download-youtube`, formData)

    if (res.data.status === 'success') {
      processLog('✅ Download do YouTube concluído!')
      status.value = 'idle'
      // Marcar como se tivéssemos feito upload
      file.value = { name: 'Video do YouTube' }
    } else {
      status.value = 'error'
      processLog(`❌ Erro: ${res.data.message}`)
    }
  } catch (e) {
    status.value = 'error'
    processLog(`❌ Erro ao baixar do YouTube: ${e.message}`)
  }
}

const startProcess = async () => {
  if (!file.value && inputMode.value === 'upload') {
    return alert('Selecione um vídeo primeiro!')
  }
  if (inputMode.value === 'youtube' && !isValidYoutubeUrl.value) {
    return alert('Por favor, baixe um vídeo do YouTube primeiro!')
  }

  logs.value = []
  currentStep.value = 0
  videoUrl.value = null
  processLog('🚀 Processo iniciado...')

  try {
    // Apenas faz upload se for modo upload (YouTube já foi baixado)
    if (inputMode.value === 'upload') {
      status.value = 'uploading'
      const formData = new FormData()
      formData.append('file', file.value)

      processLog('📤 Fazendo upload do arquivo...')
      const uploadRes = await axios.post(`${BACKEND_URL}/upload`, formData)
      if (uploadRes.data.path) {
        processLog('✅ Upload concluído!')
      }
    } else {
      processLog('✅ Usando vídeo do YouTube já baixado!')
    }

    status.value = 'processing'
    currentStep.value = 1

    processLog('⏳ Solicitando processamento ao backend...')
    const params = new FormData()
    params.append('motor', motor.value)
    params.append('encoding', encoding.value)

    const res = await axios.post(`${BACKEND_URL}/process`, params)

    if (res.data.status === 'success') {
      status.value = 'done'
      currentStep.value = 6
      videoUrl.value = `${BACKEND_URL}${res.data.video_url}`
      processLog('✨ Processamento Finalizado com Sucesso!')
    } else {
      status.value = 'error'
      processLog('❌ Erro reportado pelo backend.')
    }

  } catch (e) {
    status.value = 'error'
    processLog(`❌ Erro de Conexão ou Script: ${e.message}`)
  }
}
</script>

<template>
  <div class="container">
    <header>
      <h1>🎬 Dubbler Pro Web</h1>
      <p class="subtitle">Dublagem Automática via IA</p>
    </header>

    <div class="main-layout">
      <!-- Left Panel: Controls -->
      <div class="control-panel card">
        <h2>🛠️ Configuração</h2>

        <!-- Input Mode Selector -->
        <div class="form-group">
          <label>Fonte do Vídeo:</label>
          <div class="mode-selector">
            <button @click="inputMode = 'upload'" :class="{ active: inputMode === 'upload' }" class="mode-btn">
              📁 Upload Local
            </button>
            <button @click="inputMode = 'youtube'" :class="{ active: inputMode === 'youtube' }" class="mode-btn">
              🎥 YouTube URL
            </button>
          </div>
        </div>

        <!-- Upload Mode -->
        <div class="form-group" v-if="inputMode === 'upload'">
          <label>Arquivo de Vídeo:</label>
          <div class="file-drop-area" :class="{ 'has-file': file }">
            <input type="file" @change="handleFileChange" accept="video/*" />
            <div v-if="file" class="file-info">
              <span>📄 {{ file.name }}</span>
            </div>
            <div v-else class="placeholder">
              <span>📂 Clique ou arraste aqui</span>
            </div>
          </div>
        </div>

        <!-- YouTube Mode -->
        <div class="form-group" v-if="inputMode === 'youtube'">
          <label>URL do YouTube:</label>
          <input type="text" v-model="youtubeUrl" @input="validateYoutubeUrl" style="width: 90%;"
            placeholder="https://www.youtube.com/watch?v=..." class="youtube-input"
            :class="{ valid: isValidYoutubeUrl && youtubeUrl, invalid: !isValidYoutubeUrl && youtubeUrl }" />
          <button @click="downloadFromYoutube" :disabled="!isValidYoutubeUrl || status === 'downloading'"
            class="btn-youtube">
            <span v-if="status === 'downloading'">⬇️ Baixando...</span>
            <span v-else>⬇️ Baixar do YouTube</span>
          </button>
        </div>

        <div class="form-group">
          <label>Motor TTS:</label>
          <select v-model="motor">
            <option value="mms">MMS-TTS (Rápido/Offline)</option>
            <option value="coqui">Coqui XTTS (Clonagem de Voz)</option>
          </select>
        </div>

        <div class="form-group">
          <label>Modo de Encoding:</label>
          <select v-model="encoding">
            <option value="rapido">Rápido (GPU NVENC)</option>
            <option value="qualidade">Qualidade (CPU libx264)</option>
          </select>
        </div>

        <button @click="startProcess"
          :disabled="status === 'uploading' || status === 'downloading' || status === 'processing'" class="btn-primary"
          :class="status">
          <span v-if="status === 'uploading'">📤 Enviando...</span>
          <span v-else-if="status === 'downloading'">⬇️ Baixando...</span>
          <span v-else-if="status === 'processing'">⚙️ Processando...</span>
          <span v-else>▶️ Iniciar Dublagem</span>
        </button>
      </div>

      <!-- Right Panel: Status & Output -->
      <div class="output-panel">

        <!-- Visual Stepper -->
        <div class="stepper card">
          <div v-for="step in steps" :key="step.id" class="step-item" :class="{
            active: currentStep === step.id,
            completed: currentStep > step.id
          }">
            <div class="step-circle">
              <span class="step-icon">{{ step.icon }}</span>
            </div>
            <div class="step-label">{{ step.label }}</div>
          </div>
        </div>

        <!-- Progress Bar (Render Only) -->
        <div class="progress-container card" v-if="currentStep === 5 && progress > 0">
          <div class="progress-info">
            <span>Renderizando Vídeo Final...</span>
            <span>{{ progress }}%</span>
          </div>
          <div class="progress-bar-bg">
            <div class="progress-bar-fill" :style="{ width: progress + '%' }"></div>
          </div>
        </div>

        <!-- Terminal Logs -->
        <div class="log-window card" ref="logContainer">
          <div class="log-header">
            <span>📟 Terminal Output</span>
            <span class="badge" v-if="status === 'processing'">● LIVE</span>
          </div>
          <div class="log-content">
            <div v-for="(log, i) in logs" :key="i" class="log-line">
              <span class="arrow">></span> {{ log }}
            </div>
            <div v-if="logs.length === 0" class="log-placeholder">
              Aguardando início...
            </div>
          </div>
        </div>

        <!-- Result -->
        <div class="video-preview card success-card" v-if="videoUrl">
          <h3>✨ Resultado Disponível</h3>
          <video :src="videoUrl" controls autoplay></video>
          <a :href="videoUrl" download class="btn-download">⬇️ Baixar Vídeo Dublado</a>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
/* Scoped Styles for App.vue Components */

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

header {
  text-align: center;
  margin-bottom: 3rem;
}

.subtitle {
  color: var(--text-muted);
  margin-top: 0.5rem;
}

.main-layout {
  display: grid;
  grid-template-columns: 350px 1fr;
  gap: 2rem;
}

.card {
  background: var(--card-bg);
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
  margin-bottom: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.form-group {
  margin-bottom: 1.5rem;
}

label {
  display: block;
  margin-bottom: 0.8rem;
  color: var(--primary);
  font-weight: 600;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

/* Inputs */
select {
  width: 100%;
  background: var(--bg);
  color: white;
  border: 1px solid var(--text-muted);
  padding: 0.8rem;
  border-radius: 8px;
  font-size: 1rem;
  outline: none;
}

select:focus {
  border-color: var(--primary);
}

.file-drop-area {
  border: 2px dashed var(--text-muted);
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  position: relative;
  cursor: pointer;
  transition: all 0.3s ease;
  background: rgba(0, 0, 0, 0.2);
}

.file-drop-area:hover {
  border-color: var(--primary);
  background: rgba(122, 162, 247, 0.1);
}

.file-drop-area.has-file {
  border-color: var(--green);
  border-style: solid;
}

.file-drop-area input {
  position: absolute;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}

.file-info {
  color: var(--green);
  font-weight: bold;
}

.placeholder {
  color: var(--text-muted);
}

/* Buttons */
.btn-primary {
  width: 100%;
  background: linear-gradient(135deg, var(--primary) 0%, #5d8eea 100%);
  color: #fff;
  border: none;
  padding: 1rem;
  font-size: 1.1rem;
  font-weight: bold;
  border-radius: 8px;
  cursor: pointer;
  transition: transform 0.2s, opacity 0.2s;
  box-shadow: 0 4px 12px rgba(122, 162, 247, 0.3);
}

.btn-primary:hover {
  transform: translateY(-2px);
}

.btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  background: var(--text-muted);
  box-shadow: none;
}

/* Stepper */
.stepper {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 2rem;
  gap: 1rem;
}

.step-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  opacity: 0.3;
  transition: all 0.4s ease;
}

.step-circle {
  width: 50px;
  height: 50px;
  background: var(--bg);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 0.8rem;
  font-size: 1.5rem;
  border: 2px solid transparent;
}

.step-item.active {
  opacity: 1;
  transform: scale(1.1);
}

.step-item.active .step-circle {
  border-color: var(--primary);
  box-shadow: 0 0 15px var(--primary);
  background: var(--primary);
  color: white;
}

.step-item.completed {
  opacity: 1;
}

.step-item.completed .step-circle {
  border-color: var(--green);
  color: var(--green);
}

.step-label {
  font-size: 0.75rem;
  text-align: center;
  font-weight: 600;
}

/* Logs */
.log-window {
  height: 400px;
  display: flex;
  flex-direction: column;
  background: #0e1016;
  /* Darker black */
  border: 1px solid #333;
  padding: 0;
  overflow: hidden;
}

.log-header {
  background: #1a1b26;
  padding: 0.8rem 1rem;
  border-bottom: 1px solid #333;
  display: flex;
  justify-content: space-between;
  font-size: 0.9rem;
  font-weight: bold;
  color: var(--text-muted);
}

.log-content {
  overflow-y: auto;
  flex: 1;
  padding: 1rem;
  font-family: 'Consolas', 'Fira Code', monospace;
}

.log-line {
  padding: 3px 0;
  font-size: 0.9rem;
  color: #a9b1d6;
  border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  display: flex;
}

.arrow {
  color: var(--primary);
  margin-right: 10px;
  user-select: none;
}

.log-placeholder {
  color: var(--text-muted);
  font-style: italic;
  opacity: 0.5;
}

.badge {
  color: var(--red);
  font-weight: 900;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0% {
    opacity: 1;
  }

  50% {
    opacity: 0.5;
  }

  100% {
    opacity: 1;
  }
}

/* Result */
.success-card {
  border: 2px solid var(--green);
  text-align: center;
}

.video-preview video {
  width: 100%;
  max-width: 600px;
  border-radius: 8px;
  margin: 1.5rem 0;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}

/* Progress Bar */
.progress-container {
  padding: 1rem 1.5rem;
  animation: fadeIn 0.5s;
}

.progress-info {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
  font-weight: bold;
  color: var(--yellow);
}

.progress-bar-bg {
  width: 100%;
  height: 10px;
  background: #333;
  border-radius: 5px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--yellow), var(--green));
  transition: width 0.3s ease;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }

  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.btn-download {
  display: inline-block;
  background: var(--green);
  color: #000;
  padding: 12px 24px;
  text-decoration: none;
  font-weight: 800;
  border-radius: 8px;
  transition: transform 0.2s;
}

.btn-download:hover {
  transform: scale(1.05);
}

/* YouTube Input Styles */
.mode-selector {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.mode-btn {
  padding: 0.8rem;
  border: 2px solid var(--text-muted);
  background: var(--bg);
  color: var(--text-muted);
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.3s ease;
}

.mode-btn:hover {
  border-color: var(--primary);
  color: var(--primary);
}

.mode-btn.active {
  border-color: var(--primary);
  background: var(--primary);
  color: white;
  box-shadow: 0 4px 12px rgba(122, 162, 247, 0.3);
}

.youtube-input {
  width: 100%;
  padding: 0.8rem;
  background: var(--bg);
  border: 2px solid var(--text-muted);
  border-radius: 8px;
  color: white;
  font-size: 1rem;
  outline: none;
  transition: border-color 0.3s ease;
  margin-bottom: 1rem;
}

.youtube-input:focus {
  border-color: var(--primary);
}

.youtube-input.valid {
  border-color: var(--green);
}

.youtube-input.invalid {
  border-color: var(--red);
}

.btn-youtube {
  width: 100%;
  padding: 0.8rem;
  background: linear-gradient(135deg, #ff0000 0%, #cc0000 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.3s ease;
  margin-bottom: 1.5rem;
}

.btn-youtube:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(255, 0, 0, 0.4);
}

.btn-youtube:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

/* Mobile */
@media (max-width: 900px) {
  .main-layout {
    grid-template-columns: 1fr;
  }

  .stepper {
    overflow-x: auto;
    padding: 1rem;
  }

  .step-circle {
    width: 40px;
    height: 40px;
    font-size: 1.2rem;
  }
}
</style>
