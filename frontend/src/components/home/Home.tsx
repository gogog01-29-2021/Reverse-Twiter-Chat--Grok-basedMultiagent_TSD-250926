import { useState } from 'react';
import Chat from '../chat/Chat';
import Header from '../header/Header';
import NodeLib from '../nodeLib/NodeLib';
import FlowCanvas from '../flowCanvas/FlowCanvas';

import * as S from './styled';
import type { INode } from '../../apis/axios/nodeLib/types';
import { usePostWorkflows } from '../../apis/queries/nodeLib';
import ProjectNameModal from './ProjectNameModal';

const Home = () => {
  const [selectedNodes, setSelectedNodes] = useState<INode[]>([]);
  const [workflowTitle, setWorkflowTitle] = useState('');
  const [showProjectModal, setShowProjectModal] = useState(true);
  const [workflowId, setWorkflowId] = useState<string | null>(null);
  const { mutateAsync } = usePostWorkflows();

  console.log(selectedNodes);

  // 프로젝트명 확인 핸들러
  const handleProjectNameConfirm = (projectName: string) => {
    setWorkflowTitle(projectName);
    setShowProjectModal(false);
  };

  // 노드 선택 핸들러
  const handleNodeSelect = (node: INode) => {
    setSelectedNodes((prev) => [...prev, node]);
  };

  // 노드 제거 핸들러
  const handleNodeRemove = (index: number) => {
    setSelectedNodes((prev) => prev.filter((_, i) => i !== index));
  };

  // 전체 삭제 핸들러
  const handleClearAll = () => {
    if (window.confirm('모든 선택된 노드를 삭제하시겠습니까?')) {
      setSelectedNodes([]);
      setWorkflowId(null);
    }
  };

  // 워크플로우 불러오기 핸들러
  const handleWorkflowLoad = (workflowData: { name: string; nodes: INode[]; id?: string }) => {
    console.log('워크플로우 로드:', workflowData);
    setSelectedNodes(workflowData.nodes);
    setWorkflowTitle(workflowData.name);

    console.log(selectedNodes, '선택된노드');

    // 불러온 워크플로우에 ID가 있으면 설정
    if (workflowData.id) {
      setWorkflowId(workflowData.id);
    }
  };

  // Chat에서 워크플로우 메시지를 받았을 때 처리하는 핸들러
  const handleWorkflowReceived = (workflowData: { id: string; name: string; nodes: any[] }) => {
    console.log('Chat에서 워크플로우 수신:', workflowData);

    // 받은 워크플로우 데이터를 handleWorkflowLoad로 전달
    handleWorkflowLoad({
      id: workflowData.id,
      name: workflowData.name,
      nodes: workflowData.nodes,
    });

    // alert 제거 - 콘솔 로그로만 확인
    console.log(`워크플로우 "${workflowData.name}"이 로드되었습니다.`);
  };

  // 워크플로우 실행 핸들러
  const handleWorkflowExecute = async () => {
    const nodeIds = selectedNodes.map((node) => node.id);
    console.log('워크플로우 실행:', nodeIds);

    try {
      const result = await mutateAsync({
        name: workflowTitle,
        nodes: nodeIds,
      });
      console.log('워크플로우 생성됨:', result);

      // 생성된 워크플로우 ID를 상태에 저장
      if (result.id) {
        setWorkflowId(result.id);
        console.log('워크플로우 ID 설정됨:', result.id);
      }

      return result.id || nodeIds;
    } catch (error) {
      console.error('에러 발생:', error);
      throw error;
    }
  };

  // 프로젝트명 변경 핸들러
  const handleProjectNameChange = () => {
    setShowProjectModal(true);
  };

  // 워크플로우 타입 결정
  const conversationType = selectedNodes.length > 0 ? 'workflow' : 'general';

  return (
    <S.Wrapper>
      {/* 프로젝트명 입력 모달 */}
      <ProjectNameModal isOpen={showProjectModal} onConfirm={handleProjectNameConfirm} />

      {/* 모달이 닫혀있을 때만 메인 컨텐츠 렌더링 */}
      {!showProjectModal && (
        <>
          <Header
            onWorkflowExecute={handleWorkflowExecute}
            selectedNodesCount={selectedNodes.length}
            onWorkflowLoad={handleWorkflowLoad}
            workflowTitle={workflowTitle}
            onProjectNameEdit={handleProjectNameChange}
          />

          <S.ContantsWrapper>
            <NodeLib onNodeSelect={handleNodeSelect} />

            <S.MainContent>
              <FlowCanvas 
                selectedNodes={selectedNodes}
                onNodeRemove={handleNodeRemove}
                onClearAll={handleClearAll}
              />
            </S.MainContent>

            <S.ChatWrapper>
              {/* Chat에 워크플로우 정보 전달 및 콜백 등록 */}
              <Chat
                workflowId={workflowId}
                conversationType={conversationType}
                onWorkflowReceived={handleWorkflowReceived}
              />
            </S.ChatWrapper>
          </S.ContantsWrapper>
        </>
      )}
    </S.Wrapper>
  );
};

export default Home;
