import { instance } from '..';

export const createConversation = async () => {
  const { data } = await instance.post('/v1/conversations');
  return data;
};

export const sendMessage = async (
  conversationId: string,
  messageData: {
    parts: Array<{
      text: string;
      type: string;
    }>;
    role: string;
  },
) => {
  const { data } = await instance.post(`/v1/conversations/${conversationId}/messages`, messageData);
  return data;
};

export const getPendingMessages = async () => {
  const { data } = await instance.get('/v1/messages/pending');
  return data;
};

export const getConversation = async (conversationId: string) => {
  const { data } = await instance.get(`/v1/conversations/${conversationId}/messages`);
  return data;
};
