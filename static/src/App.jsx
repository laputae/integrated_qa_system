import React, { useState, useEffect, useRef } from 'react';

const App = () => {
  const [query, setQuery] = useState('');
  const [sourceFilter, setSourceFilter] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [answer, setAnswer] = useState('');
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const answerRef = useRef(null);

  // 学科类别（从后端获取或硬编码）
  const validSources = ['数学', '物理', '化学', '生物', '历史', '地理'];

  // 初始化会话ID
  useEffect(() => {
    fetch('http://localhost:8000/query', { method: 'POST' })
      .then(res => res.json())
      .then(data => setSessionId(data.session_id))
      .catch(err => console.error('初始化会话失败:', err));
  }, []);

  // 自动滚动到答案底部
  useEffect(() => {
    if (answerRef.current) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight;
    }
  }, [answer]);

  // 提交查询
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setAnswer('');
    const requestBody = {
      query,
      source_filter: sourceFilter || null,
      session_id: sessionId
    };

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (response.headers.get('content-type').includes('text/event-stream')) {
        // 流式响应（RAG）
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedAnswer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          const lines = chunk.split('\n\n');
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const token = line.slice(6);
              if (token) {
                accumulatedAnswer += token;
                setAnswer(accumulatedAnswer);
              }
            }
          }
        }
      } else {
        // JSON响应（MySQL）
        const data = await response.json();
        setAnswer(data.answer);
      }

      // 更新历史
      const historyRes = await fetch(`http://localhost:8000/history/${sessionId}`);
      const historyData = await historyRes.json();
      setHistory(historyData.history);
    } catch (err) {
      setAnswer('错误：无法获取答案');
      console.error('查询失败:', err);
    } finally {
      setIsLoading(false);
    }

    setQuery('');
    setSourceFilter('');
  };

  // 清除历史
  const handleClearHistory = async () => {
    try {
      await fetch(`http://localhost:8000/history/${sessionId}`, { method: 'DELETE' });
      setHistory([]);
      setAnswer('');
      alert('历史记录已清除');
    } catch (err) {
      console.error('清除历史失败:', err);
      alert('清除历史失败');
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-3xl font-bold text-center mb-6">集成问答系统</h1>
      <p className="text-center mb-4">会话ID: {sessionId || '加载中...'}</p>

      {/* 输入表单 */}
      <div className="bg-white p-6 rounded-lg shadow-md mb-6">
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700">请输入您的问题</label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              placeholder="输入问题..."
              disabled={isLoading}
            />
          </div>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700">学科类别（可选）</label>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              disabled={isLoading}
            >
              <option value="">不限</option>
              {validSources.map((source) => (
                <option key={source} value={source}>{source}</option>
              ))}
            </select>
          </div>
          <button
            type="submit"
            className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 disabled:bg-gray-400"
            disabled={isLoading}
          >
            {isLoading ? '处理中...' : '提交'}
          </button>
        </form>
      </div>

      {/* 答案显示 */}
      {answer && (
        <div className="bg-white p-6 rounded-lg shadow-md mb-6">
          <h2 className="text-xl font-semibold mb-2">答案</h2>
          <div
喧嚣
          ref={answerRef}
          className="max-h-96 overflow-y-auto prose prose-sm"
          dangerouslySetInnerHTML={{ __html: answer.replace(/\n/g, '<br>') }}
        ></div>
        </div>
      )}

      {/* 历史记录 */}
      {history.length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">最近对话历史</h2>
            <button
              onClick={handleClearHistory}
              className="text-red-600 hover:text-red-800"
            >
              清除历史
            </button>
          </div>
          <div className="space-y-4">
            {history.map((entry, idx) => (
              <div key={idx}>
                <p className="font-medium">问: {entry.question}</p>
                <p className="text-gray-600">答: {entry.answer}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

ReactDOM.createRoot(document.getElementById('root')).render(<App />);