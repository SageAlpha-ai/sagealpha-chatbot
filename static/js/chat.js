// static/js/chat.js (updated)
/* Session-aware client with auto-rename-on-first-message and avatar handling */
(async function(){
  const messagesEl = document.getElementById('messages');
  const messageBox = document.getElementById('messageBox');
  const sendBtn = document.getElementById('sendBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const chatHistoryEl = document.getElementById('chatHistory');
  const fileInput = document.getElementById('fileInput');
  const usernameEl = document.getElementById('username');
  const userEmailEl = document.getElementById('userEmail');
  const userAvatarEl = document.getElementById('userAvatar');

  let currentSessionId = null;
  let currentSessionMeta = null; // {id, title, created, message_count}

  function renderMessage(role, text, meta){
    const d = document.createElement('div');
    d.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
    if(meta && meta.filename){
      const a = document.createElement('a');
      a.href = meta.url;
      a.textContent = meta.filename;
      a.target = '_blank';
      d.appendChild(a);
      if(text) d.appendChild(document.createTextNode(' ' + text));
    } else {
      d.textContent = text;
    }
    messagesEl.appendChild(d);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  async function loadUser(){
    try{
      const r = await fetch('/user');
      if(r.ok){
        const d = await r.json();
        usernameEl.textContent = d.username || 'Guest';
        userEmailEl.textContent = d.email || '';
        // if server eventually returns avatar url, set it; otherwise use placeholder
        if(d.avatar_url){
          userAvatarEl.src = d.avatar_url;
        } else {
          // keep existing placeholder
        }
      }
    }catch(e){}
  }

  async function loadSessions(){
    try{
      const r = await fetch('/sessions');
      if(!r.ok) return;
      const j = await r.json();
      chatHistoryEl.innerHTML = '';
      const sessions = j.sessions || [];
      if(sessions.length === 0){
        const no = document.createElement('div');
        no.className = 'history-item';
        no.textContent = 'No chats yet. Click New chat to start.';
        chatHistoryEl.appendChild(no);
        return;
      }

      // render sessions in order (server returns sorted)
      sessions.forEach(s => {
        const el = document.createElement('div');
        el.className = 'history-item' + (s.id === currentSessionId ? ' active' : '');
        // show title and a smaller created time
        el.innerHTML = `<div>${escapeHtml(s.title)}</div><div class="sub">${new Date(s.created).toLocaleString()}</div>`;
        el.onclick = async () => {
          await openSession(s.id);
        };
        chatHistoryEl.appendChild(el);
      });
    }catch(e){
      console.error(e);
    }
  }

  // open session by id and render its messages
  async function openSession(sessionId){
    try{
      const r = await fetch(`/sessions/${sessionId}`);
      if(!r.ok) return;
      const j = await r.json();
      messagesEl.innerHTML = '';
      (j.session.messages || []).forEach(m => {
        renderMessage(m.role, m.content, m.meta || null);
      });
      currentSessionId = j.session.id;
      currentSessionMeta = { id: j.session.id, title: j.session.title, created: j.session.created, message_count: (j.session.messages||[]).length };
      loadSessions();
    }catch(e){ console.error(e); }
  }

  // create a new session and set as active
  newChatBtn.addEventListener('click', async () => {
    try{
      const r = await fetch('/sessions', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ title: 'New chat' })
      });
      if(!r.ok) return;
      const j = await r.json();
      currentSessionId = j.session.id;
      currentSessionMeta = { id: j.session.id, title: j.session.title, created: j.session.created, message_count: 0 };
      messagesEl.innerHTML = '';
      loadSessions();
      messageBox.focus();
    }catch(e){ console.error(e); }
  });


  // helper to escape HTML for title text
  function escapeHtml(s){
    return String(s).replace(/[&<>"']/g, (m) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]));
  }

  // when sending message: if session title is default, rename it to a short preview of first user msg
  async function maybeRenameSessionOnFirstMessage(firstMessageText){
    try{
      if(!currentSessionId) return;
      const title = (currentSessionMeta && currentSessionMeta.title) || '';
      const lower = (title || '').toLowerCase();
      if(!lower || lower.startsWith('new chat') || lower.startsWith('new conversation') || lower === 'default'){
        // create a short title: first 60 characters, remove newlines
        let t = firstMessageText.replace(/\s+/g,' ').trim().slice(0,60);
        // if string is long, trim to last full word
        if(t.length === 60){
          t = t.replace(/\s+\S*$/,'');
        }
        if(!t) t = 'Chat';
        // call rename endpoint
        const rr = await fetch(`/sessions/${currentSessionId}/rename`, {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ title: t })
        });
        if(rr.ok){
          const jj = await rr.json();
          currentSessionMeta.title = jj.session.title;
          loadSessions();
        }
      }
    }catch(e){ console.warn('rename failed', e); }
  }

  // send message function
  async function sendMessage(){
    const text = messageBox.value.trim();
    if(!text) return;
    renderMessage('user', text);
    messageBox.value = '';
    try{
      // if first message in session, rename session proactively
      if(!currentSessionId){
        // create a new session first
        const resp = await fetch('/sessions', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ title: 'New chat' }) });
        const js = await resp.json();
        currentSessionId = js.session.id;
        currentSessionMeta = { id: js.session.id, title: js.session.title, created: js.session.created, message_count: 0 };
      }
      // If session had default title, attempt rename using first message
      await maybeRenameSessionOnFirstMessage(text);

      // send message to session endpoint
      const r = await fetch('/chat_session', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ session_id: currentSessionId, message: text })
      });
      if(!r.ok){
        const err = await r.json().catch(()=>({error:'Unknown'}));
        renderMessage('assistant', '⚠ Error: '+(err.error||r.statusText));
        return;
      }
      const j = await r.json();
      renderMessage('assistant', j.response || '(no reply)');
      loadSessions();
    }catch(e){
      renderMessage('assistant', '⚠ Network error: ' + e.message);
    }
  }

  // file upload: include session_id so server will append to that session
  fileInput.addEventListener('change', async (ev) => {
    const f = ev.target.files[0];
    if(!f) return;
    const fd = new FormData();
    fd.append('file', f);
    if(currentSessionId) fd.append('session_id', currentSessionId);
    try {
      const r = await fetch('/upload', { method: 'POST', body: fd });
      if(!r.ok){ const err = await r.json().catch(()=>({error:'Upload failed'})); renderMessage('assistant', '⚠ Upload error: '+(err.error||r.statusText)); return; }
      const j = await r.json();
      renderMessage('user', '', { filename: j.filename, url: j.url });
      // optionally trigger chat response about the attachment
      const rr = await fetch('/chat_session', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ session_id: currentSessionId, message: `[attachment] ${j.url}` })});
      if(rr.ok){ const jj = await rr.json(); renderMessage('assistant', jj.response || '(no reply)'); }
      loadSessions();
    } catch(e){
      renderMessage('assistant', '⚠ Upload error: ' + e.message);
    } finally {
      fileInput.value = '';
    }
  });

  messageBox.addEventListener('keypress', (e) => {
    if(e.key === 'Enter'){ e.preventDefault(); sendMessage(); }
  });
  sendBtn.addEventListener('click', sendMessage);

  // init
  await loadUser();
  await loadSessions();
  // open newest session if exists
  try{
    const res = await fetch('/sessions');
    const js = await res.json();
    if(js.sessions && js.sessions.length > 0){
      await openSession(js.sessions[0].id);
    } else {
      const rp = await fetch('/sessions', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({title:'New chat'})});
      const newj = await rp.json();
      currentSessionId = newj.session.id;
      currentSessionMeta = { id: newj.session.id, title: newj.session.title, created: newj.session.created, message_count: 0 };
    }
  }catch(e){}
})();


// ==================== THEME TOGGLE (light/dark) ====================
(function initThemeToggle() {
  const toggle = document.getElementById('themeToggle');
  const icon = document.getElementById('themeIcon');
  if (!toggle || !icon) return;

  const DARK = 'dark';
  const LIGHT = 'light';

  function applyTheme(theme) {
    if (theme === DARK) {
      document.body.classList.add('dark-mode');
      icon.textContent = '☀️'; // show sun when in dark mode (click to go light)
    } else {
      document.body.classList.remove('dark-mode');
      icon.textContent = '🌙'; // show moon when in light mode (click to go dark)
    }
    localStorage.setItem('sagealpha-theme', theme);
  }

  // load preferred theme: saved -> system preference -> light
  const saved = localStorage.getItem('sagealpha-theme');
  if (saved === DARK || saved === LIGHT) {
    applyTheme(saved);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    applyTheme(DARK);
  } else {
    applyTheme(LIGHT);
  }

  toggle.addEventListener('click', () => {
    const isDark = document.body.classList.contains('dark-mode');
    applyTheme(isDark ? LIGHT : DARK);
  });
})();
