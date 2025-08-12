import { Route, Routes } from 'react-router';
import HomePage from './pages/home';

const Router = () => {
  return (
    <Routes>
      <Route path={'/'} element={<HomePage />} />
    </Routes>
  );
};

export default Router;
