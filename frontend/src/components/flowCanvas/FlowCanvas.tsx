import { useCallback, useMemo, useEffect } from 'react';
import {
  ReactFlow,
  addEdge,
  useNodesState,
  useEdgesState,
  Background,
  Controls,
  MiniMap,
  type Connection,
} from '@xyflow/react';
import type { INode } from '../../apis/axios/nodeLib/types';
import * as S from './styled';

interface FlowCanvasProps {
  selectedNodes: INode[];
  onNodeRemove: (index: number) => void;
  onClearAll: () => void;
}

const FlowCanvas = ({ selectedNodes, onNodeRemove, onClearAll }: FlowCanvasProps) => {
  // Convert INode to React Flow Node format
  const initialNodes = useMemo(
    () =>
      selectedNodes.map((node, index) => ({
        id: `node-${index}`,
        type: 'default',
        position: { x: index * 250 + 100, y: 100 },
        data: {
          label: node.name,
        },
        style: {
          background: '#ffffff',
          border: '2px solid #4f46e5',
          borderRadius: '8px',
          padding: '10px',
          fontSize: '14px',
          fontWeight: '500',
        },
        draggable: true,
      })),
    [selectedNodes, onNodeRemove]
  );

  // Create edges connecting nodes in sequence
  const initialEdges = useMemo(
    () =>
      selectedNodes.length > 1
        ? selectedNodes.slice(0, -1).map((_, index) => ({
            id: `edge-${index}`,
            source: `node-${index}`,
            target: `node-${index + 1}`,
            type: 'smoothstep' as const,
            animated: true,
          }))
        : [],
    [selectedNodes]
  );

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  // Update nodes and edges when selectedNodes changes
  useEffect(() => {
    setNodes(initialNodes);
    setEdges(initialEdges);
  }, [initialNodes, initialEdges, setNodes, setEdges]);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  // Debug: Always show some visual feedback
  console.log('FlowCanvas rendering with nodes:', selectedNodes.length);
  console.log('React Flow nodes:', nodes);
  console.log('React Flow edges:', edges);
  
  // Empty state when no nodes
  if (selectedNodes.length === 0) {
    return (
      <S.EmptyCanvas>
        <S.EmptyMessage>
          <S.EmptyIcon>🔧</S.EmptyIcon>
          <S.EmptyText>왼쪽에서 노드를 선택하여 워크플로우를 구성하세요</S.EmptyText>
        </S.EmptyMessage>
      </S.EmptyCanvas>
    );
  }

  return (
    <S.FlowContainer>
      <S.FlowHeader>
        <S.HeaderLeft>
          <S.HeaderTitle>워크플로우 캔버스</S.HeaderTitle>
          <S.NodeCount>{selectedNodes.length}개 노드</S.NodeCount>
        </S.HeaderLeft>
        <S.ClearButton onClick={onClearAll}>전체 삭제</S.ClearButton>
      </S.FlowHeader>

      <S.FlowWrapper>
        <div style={{ width: '100%', height: '100%', background: '#f5f5f5', border: '2px solid #4f46e5' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            fitView
            fitViewOptions={{
              padding: 0.2,
              minZoom: 0.5,
              maxZoom: 1.5,
            }}
            style={{ width: '100%', height: '100%' }}
          >
            <Background color="#f0f0f0" gap={20} />
            <Controls />
            <MiniMap
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid #ddd',
              }}
              nodeColor="#4f46e5"
              maskColor="rgba(255, 255, 255, 0.8)"
            />
          </ReactFlow>
        </div>
      </S.FlowWrapper>
    </S.FlowContainer>
  );
};

export default FlowCanvas;