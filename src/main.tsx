import ReactDOM from 'react-dom/client'
import '@fontsource-variable/manrope'
import './style.css'
import App from './App'
import { normalizeReadingUrl } from './routes'

const url = normalizeReadingUrl(new URL(window.location.href), new URL(document.baseURI))
if (url.href !== window.location.href) window.history.replaceState(null, '', url)

ReactDOM.createRoot(document.getElementById('root')!).render(<App />)
