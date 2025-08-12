import { useState, useEffect, useRef } from 'react';
import * as S from './styled';
import { instance } from '../../apis/axios';
import {
  useCreateConversation,
  useSendMessage,
  useGetPendingMessages,
  useGetConversation,
} from '../../apis/queries/chat';

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  type?: string; // 메시지 타입 추가
  workflow_id?: string; // 워크플로우 ID 추가
  workflow_data?: any; // 워크플로우 데이터 추가
}

interface ApiMessage {
  id: string;
  parts: Array<{
    text?: string;
    type: string;
    work_flow_id?: string; // work_flow 타입일 때 포함
  }>;
  role: 'user' | 'agent';
}

// Chat 컴포넌트의 Props 타입 정의
interface ChatProps {
  workflowId?: string | null;
  conversationType?: 'general' | 'workflow';
  onWorkflowReceived?: (workflowData: { id: string; name: string; nodes: any[] }) => void; // 워크플로우 수신 콜백
}

const Chat = ({ workflowId, conversationType = 'general', onWorkflowReceived }: ChatProps) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [currentMessageId, setCurrentMessageId] = useState<string | null>(null);
  const pollingIntervalRef = useRef<number | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // React Query 훅들
  const createConversationMutation = useCreateConversation();
  const sendMessageMutation = useSendMessage();
  const { refetch: refetchPendingMessages } = useGetPendingMessages();
  const { data: conversationData, refetch: refetchConversation } =
    useGetConversation(conversationId);

  // 워크플로우 상세 정보를 가져오는 함수
  const fetchWorkflowDetail = async (workflowId: string) => {
    try {
      const response = await instance.get(`/v1/workflows/${workflowId}`);
      return response.data;
    } catch (error) {
      console.error('워크플로우 정보 가져오기 실패:', error);
      throw error;
    }
  };

  // 컴포넌트 마운트 시 새 대화 세션 생성
  useEffect(() => {
    handleCreateConversation();

    // 컴포넌트 언마운트 시 폴링 정리
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  // 워크플로우 ID가 변경될 때 새 대화 세션 생성 (최초에만)
  useEffect(() => {
    if (conversationType === 'workflow' && workflowId && !conversationId) {
      handleCreateConversation();
    }
  }, [workflowId, conversationType, conversationId]);

  // input에 자동 포커스
  useEffect(() => {
    if (conversationId && inputRef.current) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  }, [conversationId]);

  // 메시지 전송 후 input에 다시 포커스
  useEffect(() => {
    if (!sendMessageMutation.isPending && inputRef.current && conversationId) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  }, [sendMessageMutation.isPending, conversationId]);

  // 처리된 워크플로우 ID를 추적하기 위한 ref
  const processedWorkflowIds = useRef<Set<string>>(new Set());

  // 대화 데이터가 업데이트되면 메시지 상태 업데이트 및 워크플로우 메시지 감지
  useEffect(() => {
    if (conversationData?.messages) {
      const convertedMessages: Message[] = conversationData.messages.map((msg: ApiMessage) => {
        // parts에서 workflow 타입 찾기
        const workflowPart = msg.parts?.find((part) => part.type === 'work_flow');
        const textPart = msg.parts?.find((part) => part.type === 'text');

        return {
          id: msg.id,
          text: textPart?.text || '',
          isUser: msg.role === 'user',
          type: workflowPart ? 'work_flow' : undefined,
          workflow_id: workflowPart?.work_flow_id,
          workflow_data: workflowPart, // 전체 워크플로우 파트 저장
        };
      });

      setMessages(convertedMessages);

      // 새로 추가된 워크플로우 메시지 감지 (아직 처리되지 않은 것만)
      const newWorkflowMessages = convertedMessages.filter(
        (msg) =>
          msg.type === 'work_flow' &&
          msg.workflow_id &&
          !msg.isUser &&
          !processedWorkflowIds.current.has(msg.workflow_id),
      );

      // 가장 최근의 워크플로우 메시지 처리
      if (newWorkflowMessages.length > 0 && onWorkflowReceived) {
        const latestWorkflowMessage = newWorkflowMessages[newWorkflowMessages.length - 1];

        console.log('새로운 워크플로우 메시지 감지:', latestWorkflowMessage);

        // 처리된 워크플로우 ID로 추가
        processedWorkflowIds.current.add(latestWorkflowMessage.workflow_id!);

        // API 호출해서 실제 워크플로우 정보 가져오기
        const loadWorkflowDetail = async () => {
          try {
            const workflowDetail = await fetchWorkflowDetail(latestWorkflowMessage.workflow_id!);

            console.log('워크플로우 상세 정보:', workflowDetail);

            // 워크플로우 데이터를 Home 컴포넌트로 전달
            onWorkflowReceived({
              id: workflowDetail.id,
              name: workflowDetail.name,
              nodes: workflowDetail.nodes || [], // API에서 받은 실제 노드 데이터
            });
          } catch (error) {
            console.error('워크플로우 정보 로드 실패:', error);

            // API 호출 실패 시 기본값으로 전달
            onWorkflowReceived({
              id: latestWorkflowMessage.workflow_id!,
              name: '워크플로우 (로드 실패)',
              nodes: [],
            });
          }
        };

        loadWorkflowDetail();
      }
    }
  }, [conversationData, onWorkflowReceived]);

  // 새 대화 세션 생성 (워크플로우 정보 포함)
  const handleCreateConversation = async () => {
    try {
      console.log('대화 세션 생성 시작...', { conversationType, workflowId });

      // 대화 생성 시 워크플로우 정보 포함
      const conversationData: any = {};

      if (conversationType === 'workflow' && workflowId) {
        conversationData.type = 'workflow';
        conversationData.workflow_id = workflowId;
      }

      const result = await createConversationMutation.mutateAsync(conversationData);
      console.log('대화 세션 생성 결과:', result);
      setConversationId(result.id);
    } catch (error) {
      console.error('대화 세션 생성 오류:', error);
      alert('대화 세션을 생성할 수 없습니다. 다시 시도해주세요.');
    }
  };

  // 펜딩 메시지 확인 폴링
  const startPollingPendingMessages = (messageId: string) => {
    console.log('폴링 시작:', messageId);
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const result = await refetchPendingMessages();
        const pendingMessages = result.data;
        console.log('펜딩 메시지 확인:', pendingMessages, '찾는 ID:', messageId);

        // 현재 메시지 ID가 펜딩 목록에 없으면 답변이 완료된 것으로 간주
        if (!pendingMessages?.pending_messages?.some((msg: any) => msg.message_id === messageId)) {
          console.log('답변 완료됨, 폴링 중단');

          // 폴링 중단
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }

          // 대화 내용 최종 업데이트
          console.log('대화 내용 최종 업데이트');
          await refetchConversation();
          setCurrentMessageId(null);
        } else {
          // 아직 펜딩 중이면 부분 응답 업데이트
          console.log('아직 펜딩 중, 부분 응답 업데이트');
          refetchConversation();
        }
      } catch (error) {
        console.error('펜딩 메시지 확인 오류:', error);
        // 에러 발생 시 폴링 중단
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
        setCurrentMessageId(null);
      }
    }, 1000);
  };

  // 메시지 전송 (워크플로우 정보 포함)
  const handleSendMessage = async () => {
    if (!input.trim() || !conversationId || sendMessageMutation.isPending) return;

    const messageText = input.trim();
    setInput('');

    try {
      console.log('메시지 전송 시작:', messageText);

      // 메시지 데이터 구성
      const messageData: any = {
        parts: [
          {
            text: messageText,
            type: 'text',
          },
        ],
        role: 'user',
      };

      // 워크플로우 타입인 경우 추가 정보 포함
      if (conversationType === 'workflow' && workflowId) {
        messageData.workflow_id = workflowId;
        messageData.context_type = 'workflow';
      }

      const result = await sendMessageMutation.mutateAsync({
        conversationId,
        messageData,
      });

      console.log('메시지 전송 결과:', result);

      if (result.status === 'accepted') {
        // 사용자 메시지를 즉시 UI에 추가
        setMessages((prev) => [
          ...prev,
          {
            id: result.message_id,
            text: messageText,
            isUser: true,
          },
        ]);

        // 현재 메시지 ID 설정 및 폴링 시작
        setCurrentMessageId(result.message_id);
        startPollingPendingMessages(result.message_id);
        refetchConversation();
      } else {
        throw new Error('메시지가 수락되지 않았습니다.');
      }
    } catch (error) {
      console.error('메시지 전송 오류:', error);
      alert('메시지 전송에 실패했습니다. 다시 시도해주세요.');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleInputClick = () => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  const isLoading =
    sendMessageMutation.isPending || createConversationMutation.isPending || !!currentMessageId;

  const getPlaceholder = () => {
    if (conversationType === 'workflow') {
      return workflowId ? '워크플로우에 대해 질문하세요...' : '워크플로우를 먼저 생성해주세요...';
    }
    return '메시지를 입력하세요...';
  };

  // 워크플로우 메시지를 다르게 렌더링하는 함수
  const renderMessage = (msg: Message) => {
    if (msg.type === 'work_flow' && !msg.isUser) {
      return (
        <S.WorkflowMessage key={msg.id}>
          <S.WorkflowHeader>🔧 워크플로우가 생성되었습니다</S.WorkflowHeader>
          <S.WorkflowContent>
            {msg.text || '워크플로우가 성공적으로 생성되었습니다.'}
          </S.WorkflowContent>
          {msg.workflow_id && <S.WorkflowId>워크플로우 ID: {msg.workflow_id}</S.WorkflowId>}
        </S.WorkflowMessage>
      );
    }

    return (
      <S.Message key={msg.id} isUser={msg.isUser}>
        {msg.text}
      </S.Message>
    );
  };

  return (
    <S.ChatContainer>
      {/* 워크플로우 모드일 때 상태 표시 */}
      {conversationType === 'workflow' && (
        <S.WorkflowIndicator>
          {workflowId ? (
            <S.WorkflowStatus>
              🔧 워크플로우 모드 (ID: {workflowId.slice(0, 8)}...)
            </S.WorkflowStatus>
          ) : (
            <S.WorkflowStatus>⚠️ 워크플로우를 먼저 생성해주세요</S.WorkflowStatus>
          )}
        </S.WorkflowIndicator>
      )}

      <S.Messages>
        {messages.map(renderMessage)}
        {isLoading && <S.Message isUser={false}>답변을 생성하고 있습니다...</S.Message>}
      </S.Messages>

      <S.InputWrapper>
        <S.Input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onClick={handleInputClick}
          placeholder={getPlaceholder()}
          disabled={
            isLoading || !conversationId || (conversationType === 'workflow' && !workflowId)
          }
          autoFocus
        />
        <S.SendButton
          onClick={handleSendMessage}
          disabled={
            isLoading ||
            !conversationId ||
            !input.trim() ||
            (conversationType === 'workflow' && !workflowId)
          }>
          {isLoading ? '전송 중...' : '전송'}
        </S.SendButton>
      </S.InputWrapper>
    </S.ChatContainer>
  );
};

export default Chat;
