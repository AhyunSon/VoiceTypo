import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite 설정.
// 마이크(getUserMedia)와 Web Speech API는 "보안 컨텍스트"에서만 동작한다.
// localhost 는 보안 컨텍스트로 취급되므로 `npm run dev` 로 접속하면 문제없다.
// (다른 기기/도메인에서 접속하려면 https 가 필요하다.)
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // 같은 네트워크의 다른 기기에서도 접속 가능하게
  },
});
