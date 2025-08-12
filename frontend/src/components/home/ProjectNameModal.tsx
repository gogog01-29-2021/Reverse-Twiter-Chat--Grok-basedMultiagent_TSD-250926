// ProjectNameModal.tsx
import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

interface ProjectNameModalProps {
  isOpen: boolean;
  onConfirm: (projectName: string) => void;
}

const ProjectNameModal = ({ isOpen, onConfirm }: ProjectNameModalProps) => {
  const [projectName, setProjectName] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (isOpen) {
      setProjectName('');
      setError('');
    }
  }, [isOpen]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setProjectName(value);

    if (error && value.trim()) {
      setError('');
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const trimmedName = projectName.trim();
    if (!trimmedName) {
      setError('프로젝트명을 입력해주세요.');
      return;
    }

    if (trimmedName.length < 2) {
      setError('프로젝트명은 2글자 이상 입력해주세요.');
      return;
    }

    onConfirm(trimmedName);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSubmit(e as any);
    }
  };

  if (!isOpen) return null;

  return (
    <ModalBackdrop>
      <ModalWrapper>
        <ModalContent>
          <WelcomeIcon>🚀</WelcomeIcon>
          <ModalTitle>새 프로젝트 시작하기</ModalTitle>
          <ModalDescription>
            프로젝트명을 입력하여 새로운 워크플로우를 시작해보세요.
          </ModalDescription>

          <form onSubmit={handleSubmit}>
            <InputWrapper>
              <InputLabel>프로젝트명</InputLabel>
              <ProjectInput
                type="text"
                value={projectName}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                placeholder="예: 배터리 열 변형 해석 프로젝트"
                autoFocus
                hasError={!!error}
              />
              {error && <ErrorMessage>{error}</ErrorMessage>}
            </InputWrapper>

            <ButtonWrapper>
              <StartButton type="submit" disabled={!projectName.trim()}>
                프로젝트 시작
              </StartButton>
            </ButtonWrapper>
          </form>

          <FooterText>언제든지 헤더에서 프로젝트명을 변경할 수 있습니다.</FooterText>
        </ModalContent>
      </ModalWrapper>
    </ModalBackdrop>
  );
};

export default ProjectNameModal;

// Styled Components
const ModalBackdrop = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(0, 0, 0, 0.9) 100%);
  backdrop-filter: blur(12px);
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
  animation: fadeIn 0.4s ease-out;

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }
`;

const ModalWrapper = styled.div`
  position: relative;
  width: 90%;
  max-width: 480px;
  animation: modalSlideIn 0.4s ease-out;

  @keyframes modalSlideIn {
    from {
      opacity: 0;
      transform: scale(0.9) translateY(-30px);
    }
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }
`;

const ModalContent = styled.div`
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border-radius: 24px;
  padding: 40px 32px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  box-shadow:
    0 25px 50px -12px rgba(0, 0, 0, 0.6),
    0 0 0 1px rgba(255, 255, 255, 0.05);
  text-align: center;
`;

const WelcomeIcon = styled.div`
  font-size: 48px;
  margin-bottom: 20px;
  animation: bounce 2s infinite;

  @keyframes bounce {
    0%,
    20%,
    50%,
    80%,
    100% {
      transform: translateY(0);
    }
    40% {
      transform: translateY(-10px);
    }
    60% {
      transform: translateY(-5px);
    }
  }
`;

const ModalTitle = styled.h1`
  color: #f8fafc;
  font-size: 28px;
  font-weight: 700;
  margin: 0 0 12px 0;
  background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const ModalDescription = styled.p`
  color: #94a3b8;
  font-size: 16px;
  line-height: 1.6;
  margin: 0 0 32px 0;
`;

const InputWrapper = styled.div`
  margin-bottom: 24px;
  text-align: left;
`;

const InputLabel = styled.label`
  display: block;
  color: #f1f5f9;
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 8px;
`;

const ProjectInput = styled.input<{ hasError: boolean }>`
  width: 100%;
  padding: 14px 16px;
  border-radius: 12px;
  border: 2px solid ${({ hasError }) => (hasError ? '#ef4444' : 'rgba(148, 163, 184, 0.3)')};
  background: rgba(30, 41, 59, 0.5);
  color: #f8fafc;
  font-size: 16px;
  transition: all 0.3s ease;
  box-sizing: border-box;

  &::placeholder {
    color: #64748b;
  }

  &:focus {
    outline: none;
    border-color: ${({ hasError }) => (hasError ? '#ef4444' : '#0068fe')};
    background: rgba(30, 41, 59, 0.8);
    box-shadow: 0 0 0 3px
      ${({ hasError }) => (hasError ? 'rgba(239, 68, 68, 0.2)' : 'rgba(0, 104, 254, 0.2)')};
  }

  &:hover {
    border-color: ${({ hasError }) => (hasError ? '#ef4444' : 'rgba(148, 163, 184, 0.5)')};
  }
`;

const ErrorMessage = styled.div`
  color: #ef4444;
  font-size: 14px;
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 4px;

  &::before {
    content: '⚠️';
    font-size: 12px;
  }
`;

const ButtonWrapper = styled.div`
  margin-bottom: 20px;
`;

const StartButton = styled.button<{ disabled?: boolean }>`
  width: 100%;
  padding: 16px 24px;
  border-radius: 12px;
  border: none;
  background: ${({ disabled }) =>
    disabled ? 'rgba(148, 163, 184, 0.3)' : 'linear-gradient(135deg, #0068fe 0%, #0052cc 100%)'};
  color: ${({ disabled }) => (disabled ? '#64748b' : '#fff')};
  font-size: 16px;
  font-weight: 600;
  cursor: ${({ disabled }) => (disabled ? 'not-allowed' : 'pointer')};
  transition: all 0.3s ease;
  box-shadow: ${({ disabled }) => (disabled ? 'none' : '0 8px 25px rgba(0, 104, 254, 0.3)')};

  &:hover {
    ${({ disabled }) =>
      !disabled &&
      `
      background: linear-gradient(135deg, #0052cc 0%, #003d99 100%);
      transform: translateY(-2px);
      box-shadow: 0 12px 35px rgba(0, 104, 254, 0.4);
    `}
  }

  &:active {
    ${({ disabled }) =>
      !disabled &&
      `
      transform: translateY(0);
      box-shadow: 0 4px 15px rgba(0, 104, 254, 0.3);
    `}
  }
`;

const FooterText = styled.p`
  color: #64748b;
  font-size: 12px;
  margin: 0;
  line-height: 1.4;
`;
