import { instance } from '..';

export const getToken = async () => {
  const res = await instance.post('/v1/auth/token');
  return res.data;
};
