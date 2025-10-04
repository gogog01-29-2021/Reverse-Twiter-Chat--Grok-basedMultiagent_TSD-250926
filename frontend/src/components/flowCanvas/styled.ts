import styled from 'styled-components';

export const FlowContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 500px;
  background: #fafafa;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #e0e0e0;
  margin: 8px;
  flex: 1;
`;

export const FlowHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  background: #ffffff;
  border-bottom: 1px solid #e0e0e0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
`;

export const HeaderLeft = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

export const HeaderTitle = styled.h3`
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #333;
`;

export const NodeCount = styled.span`
  padding: 4px 8px;
  background: #4f46e5;
  color: white;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
`;

export const ClearButton = styled.button`
  padding: 8px 16px;
  background: #ef4444;
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 500;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background: #dc2626;
  }
`;

export const FlowWrapper = styled.div`
  flex: 1;
  height: 100%;
  min-height: 400px;

  .react-flow__node {
    background: #ffffff;
    border: 2px solid #4f46e5;
    border-radius: 8px;
    padding: 0;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    transition: all 0.2s;

    &:hover {
      box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
      transform: translateY(-2px);
    }

    &.selected {
      border-color: #6366f1;
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
    }
  }

  .react-flow__edge {
    stroke: #4f46e5;
    stroke-width: 2;
  }

  .react-flow__edge.animated {
    stroke-dasharray: 5;
    animation: dashdraw 0.5s linear infinite;
  }

  .react-flow__edge.selected {
    stroke: #6366f1;
  }

  .react-flow__controls {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
  }

  .react-flow__controls button {
    background: white;
    border: none;
    border-bottom: 1px solid #e0e0e0;
    cursor: pointer;
    transition: background-color 0.2s;

    &:hover {
      background: #f5f5f5;
    }

    &:last-child {
      border-bottom: none;
    }
  }

  @keyframes dashdraw {
    to {
      stroke-dashoffset: -10;
    }
  }
`;

export const NodeContent = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  min-width: 180px;
  background: white;
  border-radius: 6px;
`;

export const NodeLabel = styled.span`
  font-size: 14px;
  font-weight: 500;
  color: #333;
  flex: 1;
  text-align: left;
`;

export const RemoveButton = styled.button`
  background: #ef4444;
  color: white;
  border: none;
  border-radius: 50%;
  width: 20px;
  height: 20px;
  font-size: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-left: 8px;
  transition: background-color 0.2s;

  &:hover {
    background: #dc2626;
  }
`;

export const EmptyCanvas = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  min-height: 500px;
  background: #fafafa;
  border-radius: 8px;
  border: 2px dashed #d1d5db;
  margin: 8px;
  flex: 1;
`;

export const EmptyMessage = styled.div`
  text-align: center;
  color: #6b7280;
`;

export const EmptyIcon = styled.div`
  font-size: 48px;
  margin-bottom: 16px;
`;

export const EmptyText = styled.p`
  margin: 0;
  font-size: 16px;
  font-weight: 500;
`;