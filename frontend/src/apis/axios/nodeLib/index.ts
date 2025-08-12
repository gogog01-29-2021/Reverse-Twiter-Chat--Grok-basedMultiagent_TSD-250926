import { instance } from '..';

export const getNodes = async () => {
  const { data } = await instance.get('/v1/nodes');
  return data;
};

export const postWorkflows = async (data: any) => {
  const res = await instance.post('/v1/workflows/', data);

  return res.data;
};
