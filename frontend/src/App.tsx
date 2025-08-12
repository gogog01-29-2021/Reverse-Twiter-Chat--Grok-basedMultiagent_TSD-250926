import { BrowserRouter } from 'react-router';
import MainProvider from './lib/providers/Main';
import Router from './Router';
import { Suspense } from 'react';

function App() {
  return (
    <MainProvider>
      <BrowserRouter>
        <Suspense fallback={<p>Loading...</p>}>
          <Router />
        </Suspense>
      </BrowserRouter>
    </MainProvider>
  );
}

export default App;
