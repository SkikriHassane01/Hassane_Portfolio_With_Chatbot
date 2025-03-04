import React, { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import gsap from 'gsap';
import skillsData from '../../utils/skillsData';

const TechSphere = () => {
  const mountRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedSkill, setSelectedSkill] = useState(null);

  useEffect(() => {
    let scene, camera, renderer, cardGroup;
    let animationFrameId;
    let controls = { isMouseDown: false, startX: 0, startY: 0 };
    let raycaster, mouse;
    let cards = [];
    let autoRotationSpeed = 0.001;
    
    const textureLoader = new THREE.TextureLoader();
    textureLoader.crossOrigin = 'anonymous';

    const loadTexture = (skill) => {
      return new Promise((resolve) => {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');

        // Load skill icon
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          // Clear canvas with transparent background
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          // No background fill - completely transparent
          
          // Draw rounded border only (no background)
          const radius = 40;
          ctx.beginPath();
          ctx.moveTo(radius, 0);
          ctx.lineTo(canvas.width - radius, 0);
          ctx.quadraticCurveTo(canvas.width, 0, canvas.width, radius);
          ctx.lineTo(canvas.width, canvas.height - radius);
          ctx.quadraticCurveTo(canvas.width, canvas.height, canvas.width - radius, canvas.height);
          ctx.lineTo(radius, canvas.height);
          ctx.quadraticCurveTo(0, canvas.height, 0, canvas.height - radius);
          ctx.lineTo(0, radius);
          ctx.quadraticCurveTo(0, 0, radius, 0);
          ctx.closePath();
          
          // Add glowing border
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
          ctx.lineWidth = 3;
          ctx.stroke();

          // Draw skill level in top right corner
          const levelSize = 60;
          const levelX = canvas.width - levelSize - 20;
          const levelY = 20;
          
          // Level background circle with subtle glow
          ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
          ctx.beginPath();
          ctx.arc(levelX + levelSize/2, levelY + levelSize/2, levelSize/2 + 3, 0, Math.PI * 2);
          ctx.fill();
          
          // Level border
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(levelX + levelSize/2, levelY + levelSize/2, levelSize/2, 0, Math.PI * 2);
          ctx.stroke();
          
          // Level text
          ctx.font = 'bold 32px Arial';
          ctx.fillStyle = '#CDDC39';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(skill.level + '%', levelX + levelSize/2, levelY + levelSize/2);

          // Draw icon with improved quality
          const iconSize = canvas.width * 0.45;
          const x = (canvas.width - iconSize) / 2;
          const y = (canvas.height - iconSize) / 2 - 10;
          
          // Add glow effect to icon
          ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
          ctx.shadowBlur = 20;
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 0;
          ctx.drawImage(img, x, y, iconSize, iconSize);
          ctx.shadowColor = 'transparent';

          // Draw skill name with better visibility
          ctx.font = 'bold 36px Arial';
          ctx.fillStyle = '#ffffff';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          
          // Add text shadow for better readability
          ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
          ctx.shadowBlur = 8;
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 0;
          ctx.fillText(skill.name, canvas.width/2, canvas.height - 60);
          ctx.shadowColor = 'transparent';

          // Draw progress bar at bottom
          const barWidth = canvas.width * 0.7;
          const barHeight = 8;
          const barX = (canvas.width - barWidth) / 2;
          const barY = canvas.height - 25;

          // Bar background with rounded corners
          ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
          ctx.beginPath();
          ctx.roundRect(barX, barY, barWidth, barHeight, 4);
          ctx.fill();

          // Progress with rounded corners and gradient
          const progressGradient = ctx.createLinearGradient(barX, 0, barX + barWidth, 0);
          progressGradient.addColorStop(0, '#9CCC65');
          progressGradient.addColorStop(1, '#CDDC39');
          ctx.fillStyle = progressGradient;
          ctx.beginPath();
          ctx.roundRect(barX, barY, barWidth * (skill.level/100), barHeight, 4);
          ctx.fill();

          const texture = new THREE.CanvasTexture(canvas);
          texture.needsUpdate = true;
          texture.anisotropy = 16; // Improve texture quality
          resolve(texture);
        };
        img.onerror = () => {
          // Handle image loading error
          console.error(`Failed to load image: ${skill.icon}`);
          
          // Create fallback texture with just the name
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          
          // No background fill - completely transparent
          
          // Draw rounded border only
          const radius = 40;
          ctx.beginPath();
          ctx.moveTo(radius, 0);
          ctx.lineTo(canvas.width - radius, 0);
          ctx.quadraticCurveTo(canvas.width, 0, canvas.width, radius);
          ctx.lineTo(canvas.width, canvas.height - radius);
          ctx.quadraticCurveTo(canvas.width, canvas.height, canvas.width - radius, canvas.height);
          ctx.lineTo(radius, canvas.height);
          ctx.quadraticCurveTo(0, canvas.height, 0, canvas.height - radius);
          ctx.lineTo(0, radius);
          ctx.quadraticCurveTo(0, 0, radius, 0);
          ctx.closePath();
          
          // Add border
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
          ctx.lineWidth = 3;
          ctx.stroke();
          
          // Draw first letter as fallback with glow
          ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
          ctx.shadowBlur = 20;
          ctx.font = 'bold 120px Arial';
          ctx.fillStyle = '#ffffff';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(skill.name.charAt(0), canvas.width/2, canvas.height/2 - 20);
          ctx.shadowColor = 'transparent';
          
          // Draw skill name
          ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
          ctx.shadowBlur = 8;
          ctx.font = 'bold 36px Arial';
          ctx.fillText(skill.name, canvas.width/2, canvas.height - 60);
          ctx.shadowColor = 'transparent';
          
          const texture = new THREE.CanvasTexture(canvas);
          texture.needsUpdate = true;
          texture.anisotropy = 16;
          resolve(texture);
        };
        img.src = skill.icon;
      });
    };

    const createCard = async (skill, position, index) => {
      const texture = await loadTexture(skill);
      const geometry = new THREE.PlaneGeometry(80, 80);
      
      // Create materials with improved settings for better visibility
      const material = new THREE.MeshPhysicalMaterial({
        map: texture,
        transparent: true,
        side: THREE.DoubleSide,
        metalness: 0.0, // Reduced metalness for less reflection
        roughness: 0.2, // Reduced roughness for better light reflection
        emissive: new THREE.Color(0x666666), // Brighter emissive color
        emissiveIntensity: 0.4, // Increased emissive intensity
        envMapIntensity: 1.5, // Enhance environment map effect
        clearcoat: 0.2, // Reduced clearcoat for less shine
        clearcoatRoughness: 0.2
      });

      const card = new THREE.Mesh(geometry, material);
      card.position.copy(position);
      card.lookAt(new THREE.Vector3(0, 0, 0));
      card.rotateY(Math.PI);

      card.userData = {
        skill,
        index,
        originalPosition: position.clone(),
        originalRotation: card.rotation.clone(),
        originalScale: new THREE.Vector3(1, 1, 1),
        hovered: false,
        selected: false
      };

      return card;
    };

    const createSkillSphere = async () => {
      const radius = 250;
      const phi = Math.PI * (3 - Math.sqrt(5));
      const positions = [];

      // Calculate positions
      for (let i = 0; i < skillsData.length; i++) {
        const y = 1 - (i / (skillsData.length - 1)) * 2;
        const radiusAtY = Math.sqrt(1 - y * y);
        const theta = phi * i;

        positions.push(new THREE.Vector3(
          Math.cos(theta) * radiusAtY * radius,
          y * radius,
          Math.sin(theta) * radiusAtY * radius
        ));
      }

      // Create cards
      for (let i = 0; i < skillsData.length; i++) {
        const card = await createCard(skillsData[i], positions[i], i);
        cards.push(card);
        cardGroup.add(card);
      }
    };

    const handleClick = (event) => {
      event.preventDefault();
      
      const rect = mountRef.current.getBoundingClientRect();
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(cards);

      if (intersects.length > 0) {
        const clickedCard = intersects[0].object;
        
        // If clicking the same card, deselect it
        if (clickedCard.userData.selected) {
          clickedCard.userData.selected = false;
          
          // Simplified animation - just scale back to normal with a subtle effect
          gsap.to(clickedCard.scale, {
            x: 1, y: 1, z: 1,
            duration: 0.4,
            ease: "back.out(1.2)"
          });
          gsap.to(clickedCard.position, {
            x: clickedCard.userData.originalPosition.x,
            y: clickedCard.userData.originalPosition.y,
            z: clickedCard.userData.originalPosition.z,
            duration: 0.4,
            ease: "power2.out"
          });
          gsap.to(clickedCard.material, {
            emissiveIntensity: 0.1,
            duration: 0.3
          });
          
          // Animate other cards back to normal
          cards.forEach(card => {
            if (card !== clickedCard) {
              gsap.to(card.scale, {
                x: 1, y: 1, z: 1,
                duration: 0.4,
                ease: "power2.out"
              });
              gsap.to(card.material, {
                emissiveIntensity: 0.1,
                duration: 0.3
              });
            }
          });
          
          // Use a slight delay to make the transition smoother when hiding the info panel
          setTimeout(() => {
            setSelectedSkill(null);
          }, 100);
        } else {
          // Deselect previously selected card
          cards.forEach(card => {
            if (card.userData.selected) {
              card.userData.selected = false;
              
              // Simplified transition for previously selected card
              gsap.to(card.scale, {
                x: 0.9, y: 0.9, z: 0.9,
                duration: 0.3,
                ease: "power2.out"
              });
              gsap.to(card.position, {
                x: card.userData.originalPosition.x,
                y: card.userData.originalPosition.y,
                z: card.userData.originalPosition.z,
                duration: 0.3,
                ease: "power2.out"
              });
              gsap.to(card.material, {
                emissiveIntensity: 0.05,
                duration: 0.2
              });
            }
          });
          
          // Select new card
          clickedCard.userData.selected = true;
          
          // First update the state to show info panel immediately
          setSelectedSkill(clickedCard.userData.skill);
          
          // Simplified animations for selected card - more subtle movement
          cards.forEach(card => {
            if (card === clickedCard) {
              // Move selected card forward with a more subtle animation
              const direction = new THREE.Vector3(0, 0, -30).applyQuaternion(camera.quaternion);
              const targetPosition = new THREE.Vector3().copy(card.userData.originalPosition).add(direction);
              
              // Simple scale animation - no complex sequence
              gsap.to(card.scale, {
                x: 1.2, y: 1.2, z: 1.2,
                duration: 0.4,
                ease: "back.out(1.2)"
              });
              
              // Simpler position animation
              gsap.to(card.position, {
                x: targetPosition.x,
                y: targetPosition.y,
                z: targetPosition.z,
                duration: 0.4,
                ease: "power2.out"
              });
              
              // Material animation for subtle glow effect
              gsap.to(card.material, {
                emissiveIntensity: 0.4,
                duration: 0.3
              });
              
              // Very subtle rotation
              gsap.to(card.rotation, {
                x: card.userData.originalRotation.x + 0.02,
                y: card.userData.originalRotation.y + 0.02,
                z: card.userData.originalRotation.z,
                duration: 0.4,
                ease: "power2.out"
              });
            } else {
              // Shrink other cards with simpler animation
              gsap.to(card.scale, {
                x: 0.8, y: 0.8, z: 0.8,
                duration: 0.3,
                ease: "power2.out"
              });
              gsap.to(card.material, {
                emissiveIntensity: 0.05,
                duration: 0.3
              });
            }
          });
        }
      } else {
        // Clicked outside any card, deselect all
        let hadSelection = false;
        cards.forEach(card => {
          if (card.userData.selected) {
            card.userData.selected = false;
            hadSelection = true;
          }
        });
        
        if (hadSelection) {
          // Reset all cards with simplified animation
          cards.forEach(card => {
            gsap.to(card.scale, {
              x: 1, y: 1, z: 1,
              duration: 0.4,
              ease: "back.out(1.2)"
            });
            gsap.to(card.position, {
              x: card.userData.originalPosition.x,
              y: card.userData.originalPosition.y,
              z: card.userData.originalPosition.z,
              duration: 0.4,
              ease: "power2.out"
            });
            gsap.to(card.material, {
              emissiveIntensity: 0.1,
              duration: 0.3
            });
          });
          
          // Use a slight delay to make the transition smoother when hiding the info panel
          setTimeout(() => {
            setSelectedSkill(null);
          }, 100);
        }
      }
    };

    const init = async () => {
      if (!mountRef.current) return;

      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;

      scene = new THREE.Scene();
      // Make the scene background completely transparent
      scene.background = null;

      camera = new THREE.PerspectiveCamera(60, width / height, 1, 1000);
      camera.position.z = 500;

      renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true, // Enable alpha for transparent background
        powerPreference: "high-performance"
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.setClearColor(0x000000, 0); // Set clear color with 0 alpha (transparent)
      mountRef.current.appendChild(renderer.domElement);

      // Setup raycaster
      raycaster = new THREE.Raycaster();
      mouse = new THREE.Vector2();

      // Create card group
      cardGroup = new THREE.Group();
      scene.add(cardGroup);

      // Improved lighting setup
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.8); // Increased intensity
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5); // Increased intensity
      directionalLight.position.set(1, 1, 1);
      scene.add(directionalLight);
      
      // Add point lights for better highlights
      const pointLight1 = new THREE.PointLight(0x3366ff, 0.8, 1000); // Increased intensity
      pointLight1.position.set(200, 100, 300);
      scene.add(pointLight1);
      
      const pointLight2 = new THREE.PointLight(0x33ff77, 0.6, 1000); // Increased intensity
      pointLight2.position.set(-200, -100, 300);
      scene.add(pointLight2);

      // Add environment map for better reflections
      const cubeTextureLoader = new THREE.CubeTextureLoader();
      const envMap = cubeTextureLoader.load([
        'https://threejs.org/examples/textures/cube/Bridge2/posx.jpg',
        'https://threejs.org/examples/textures/cube/Bridge2/negx.jpg',
        'https://threejs.org/examples/textures/cube/Bridge2/posy.jpg',
        'https://threejs.org/examples/textures/cube/Bridge2/negy.jpg',
        'https://threejs.org/examples/textures/cube/Bridge2/posz.jpg',
        'https://threejs.org/examples/textures/cube/Bridge2/negz.jpg'
      ]);
      scene.environment = envMap;

      // Create skills sphere
      await createSkillSphere();
      setIsLoading(false);

      // Add event listeners
      mountRef.current.addEventListener('click', handleClick);
      mountRef.current.addEventListener('mousedown', handleMouseDown);
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      mountRef.current.addEventListener('wheel', handleMouseWheel);
      mountRef.current.addEventListener('touchstart', handleTouchStart);
      window.addEventListener('touchmove', handleTouchMove);
      window.addEventListener('touchend', handleTouchEnd);
      window.addEventListener('resize', handleResize);
    };

    const handleMouseDown = (event) => {
      controls.isMouseDown = true;
      controls.startX = event.clientX;
      controls.startY = event.clientY;
      autoRotationSpeed = 0;
      mountRef.current.style.cursor = 'grabbing';
    };

    const handleMouseMove = (event) => {
      if (controls.isMouseDown) {
        const deltaX = event.clientX - controls.startX;
        const deltaY = event.clientY - controls.startY;
        cardGroup.rotation.y += deltaX * 0.005;
        cardGroup.rotation.x += deltaY * 0.005;
        controls.startX = event.clientX;
        controls.startY = event.clientY;
      } else {
        // Handle hover effect when not dragging
        const rect = mountRef.current.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(cards);

        // Reset all hover states
        cards.forEach(card => {
          if (!card.userData.selected) {
            card.userData.hovered = false;
          }
        });

        // Set hover state for intersected card
        if (intersects.length > 0 && !controls.isMouseDown) {
          const hoveredCard = intersects[0].object;
          if (!hoveredCard.userData.selected) {
            hoveredCard.userData.hovered = true;
            mountRef.current.style.cursor = 'pointer';
          }
        } else if (!controls.isMouseDown) {
          mountRef.current.style.cursor = 'grab';
        }
      }
    };

    const handleMouseUp = () => {
      controls.isMouseDown = false;
      autoRotationSpeed = 0.001;
      mountRef.current.style.cursor = 'grab';
    };

    const handleMouseWheel = (event) => {
      event.preventDefault();
      camera.position.z = Math.max(300, Math.min(800, camera.position.z + event.deltaY * 0.5));
    };

    const handleTouchStart = (event) => {
      if (event.touches.length === 1) {
        controls.isMouseDown = true;
        controls.startX = event.touches[0].clientX;
        controls.startY = event.touches[0].clientY;
        autoRotationSpeed = 0;
      }
    };

    const handleTouchMove = (event) => {
      if (controls.isMouseDown && event.touches.length === 1) {
        const deltaX = event.touches[0].clientX - controls.startX;
        const deltaY = event.touches[0].clientY - controls.startY;
        cardGroup.rotation.y += deltaX * 0.005;
        cardGroup.rotation.x += deltaY * 0.005;
        controls.startX = event.touches[0].clientX;
        controls.startY = event.touches[0].clientY;
      }
    };

    const handleTouchEnd = () => {
      controls.isMouseDown = false;
      autoRotationSpeed = 0.001;
    };

    const handleResize = () => {
      if (!mountRef.current) return;
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    const animate = () => {
      // Update card animations based on hover/selected state
      cards.forEach(card => {
        if (card.userData.hovered && !card.userData.selected) {
          // Smooth hover animation
          card.scale.lerp(new THREE.Vector3(1.15, 1.15, 1.15), 0.1);
          card.material.emissiveIntensity = THREE.MathUtils.lerp(
            card.material.emissiveIntensity, 
            0.3, 
            0.1
          );
        } 
        else if (!card.userData.selected && !card.userData.hovered) {
          // Return to normal state
          card.scale.lerp(card.userData.originalScale, 0.1);
          card.material.emissiveIntensity = THREE.MathUtils.lerp(
            card.material.emissiveIntensity, 
            0.1, 
            0.1
          );
        }
      });
      
      if (!controls.isMouseDown) {
        cardGroup.rotation.y += autoRotationSpeed;
      }
      renderer.render(scene, camera);
      animationFrameId = requestAnimationFrame(animate);
    };

    init();
    animate();

    return () => {
      cancelAnimationFrame(animationFrameId);
      
      if (mountRef.current) {
        mountRef.current.removeEventListener('click', handleClick);
        mountRef.current.removeEventListener('mousedown', handleMouseDown);
        mountRef.current.removeEventListener('wheel', handleMouseWheel);
        mountRef.current.removeEventListener('touchstart', handleTouchStart);
      }
      
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleTouchEnd);
      window.removeEventListener('resize', handleResize);

      if (mountRef.current && renderer) {
        mountRef.current.removeChild(renderer.domElement);
      }

      if (cardGroup) {
        cardGroup.traverse(child => {
          if (child instanceof THREE.Mesh) {
            child.geometry.dispose();
            if (child.material.map) {
              child.material.map.dispose();
            }
            child.material.dispose();
          }
        });
      }

      if (renderer) {
        renderer.dispose();
      }
    };
  }, []);

  return (
    <section id="techSphere" className="py-20 bg-gradient-to-b from-gray-900 to-black">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl font-bold text-white">
            My Technology Stack
          </h2>
          <p className="mt-4 text-lg text-gray-300 max-w-2xl mx-auto">
            Drag to rotate and explore the technologies I work with
          </p>
        </motion.div>
        
        <div className="relative h-[600px] md:h-[700px] rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(66,153,225,0.2)] border border-blue-900/30">
          <div ref={mountRef} className="w-full h-full cursor-grab" />
          
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <div className="animate-spin h-10 w-10 border-4 border-blue-500 rounded-full border-t-transparent"></div>
            </div>
          )}
          
          {selectedSkill && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              transition={{ 
                duration: 0.4,
                ease: "easeOut"
              }}
              className="absolute bottom-4 right-4 p-5 bg-black/80 backdrop-blur-md rounded-lg text-white max-w-xs shadow-lg border border-gray-700/50"
              style={{
                boxShadow: "0 0 20px rgba(0, 0, 0, 0.5), 0 0 30px rgba(66, 153, 225, 0.2)"
              }}
            >
              <div className="flex items-center gap-4 mb-3">
                <div className="w-12 h-12 flex items-center justify-center bg-gray-800/70 rounded-full p-2 shadow-inner">
                  <img 
                    src={selectedSkill.icon} 
                    alt={selectedSkill.name} 
                    className="w-8 h-8 object-contain"
                    onError={(e) => {
                      e.target.onerror = null;
                      e.target.src = "https://via.placeholder.com/40?text=" + selectedSkill.name.charAt(0);
                    }}
                  />
                </div>
                <h3 className="text-xl font-bold text-white">{selectedSkill.name}</h3>
              </div>
              <div className="mb-1">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm text-gray-300">Proficiency</span>
                  <span className="text-sm font-medium text-green-400">{selectedSkill.level}%</span>
                </div>
                <div className="w-full bg-gray-700/70 rounded-full h-2.5 overflow-hidden">
                  <div 
                    className="h-2.5 rounded-full bg-gradient-to-r from-green-500 to-lime-400"
                    style={{ 
                      width: `${selectedSkill.level}%`,
                      transition: "width 0.5s ease-out"
                    }}
                  />
                </div>
              </div>
            </motion.div>
          )}
          
          <div className="absolute bottom-4 left-4 p-2 bg-black/70 backdrop-blur-sm rounded-lg text-sm text-gray-300">
            <p>Drag to rotate | Scroll to zoom | Click to select</p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default TechSphere;