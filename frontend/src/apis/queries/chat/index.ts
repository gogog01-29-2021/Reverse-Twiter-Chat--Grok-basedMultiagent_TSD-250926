import { useMutation, useQuery } from '@tanstack/react-query';
import {
  createConversation,
  sendMessage,
  getPendingMessages,
  getConversation,
} from '../../axios/chat';

export const useCreateConversation = () => {
  return useMutation({
    mutationFn: () => createConversation(),
  });
};

export const useSendMessage = () => {
  return useMutation({
    mutationFn: (params: {
      conversationId: string;
      messageData: {
        parts: Array<{
          text: string;
          type: string;
        }>;
        role: string;
      };
    }) => sendMessage(params.conversationId, params.messageData),
  });
};

export const useGetPendingMessages = () => {
  return useQuery({
    queryKey: ['pendingMessages'],
    queryFn: () => getPendingMessages(),
    enabled: false,
    refetchInterval: false,
  });
};

export const useGetConversation = (conversationId: string | null) => {
  return useQuery({
    queryKey: ['conversation', conversationId],
    queryFn: () => getConversation(conversationId!),
    enabled: !!conversationId,
  });
};
