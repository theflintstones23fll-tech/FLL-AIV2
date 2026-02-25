document.addEventListener('DOMContentLoaded', function() {
    initNavigation();
    initModals();
    initAuthForms();
    initUploadForms();
    initSearch();
    initDatabaseControls();
    initArtifactDetails();
    initArtifactSelection();
    checkAuth();
    initSmoothScrolling();
});

function initArtifactSelection() {
    const checkboxes = document.querySelectorAll('.artifact-select');
    const compareBtn = document.getElementById('compareSelectedBtn');
    const selectionCount = document.getElementById('selectionCount');
    
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const selected = document.querySelectorAll('.artifact-select:checked');
            const count = selected.length;
            
            if (selectionCount) {
                selectionCount.textContent = count;
            }
            
            if (compareBtn) {
                compareBtn.disabled = count < 2;
                
                document.querySelectorAll('.artifact-card').forEach(card => {
                    card.classList.remove('selected-for-compare');
                });
                
                selected.forEach(cb => {
                    const card = cb.closest('.artifact-card');
                    if (cb.checked) {
                        card.classList.add('selected-for-compare');
                    }
                });
            }
        });
    });
    
    if (compareBtn) {
        compareBtn.addEventListener('click', async function() {
            const selected = document.querySelectorAll('.artifact-select:checked');
            if (selected.length < 2) return;
            
            const ids = Array.from(selected).map(cb => parseInt(cb.getAttribute('data-id')));
            await compareMultipleArtifacts(ids);
        });
    }
}

async function compareMultipleArtifacts(ids) {
    showLoading('Analyzing puzzle connections...');
    
    try {
        const response = await fetch('/compare_multiple_artifacts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ artifact_ids: ids })
        });
        
        const result = await response.json();
        hideLoading();
        
        if (result.error) {
            showError(result.error, 'resultsContainer');
            return;
        }
        
        displayPuzzleComparison(result);
        
    } catch (error) {
        hideLoading();
        showError(`Comparison failed: ${error.message}`, 'resultsContainer');
    }
}

function displayPuzzleComparison(data) {
    const grid = document.getElementById('selectedArtifactsGrid');
    
    let artifactsHtml = data.artifacts.map((art, idx) => {
        const isDisrupting = data.disrupting_pieces && data.disrupting_pieces.includes(idx);
        const disruptingClass = isDisrupting ? 'disrupting' : '';
        return `
            <div class="puzzle-artifact-item ${disruptingClass}">
                <img src="${art.image_base64}" alt="${art.name}">
                <span class="artifact-num">#${idx + 1}</span>
                <p>${art.name}</p>
                ${isDisrupting ? '<span class="disrupt-badge">Disrupts!</span>' : ''}
            </div>
        `;
    }).join('');
    
    let arrowsHtml = '';
    data.pair_scores.forEach((pair, idx) => {
        const canConnect = pair.can_connect;
        arrowsHtml += `
            <div class="connection-arrow ${canConnect ? 'can-connect' : 'cannot-connect'}">
                <i class="fas fa-arrow-${canConnect ? 'right' : 'times'}"></i>
                <span>${pair.score.toFixed(0)}%</span>
            </div>
        `;
    });
    
    grid.innerHTML = artifactsHtml + '<div class="connection-arrows">' + arrowsHtml + '</div>';
    
    const assemblyImagesContainer = document.getElementById('assemblyImages');
    
    if (data.assembly_images && data.assembly_images.some(img => img)) {
        assemblyImagesContainer.innerHTML = data.assembly_images.map((img, idx) => {
            if (!img) return '';
            const pair = data.pair_scores[idx];
            return `
                <div class="connection-pair-view">
                    <div class="pair-header">
                        <span class="pair-label">Connection #${idx + 1}: Piece ${idx + 1} → Piece ${(idx + 1) % data.artifacts.length + 1}</span>
                        <span class="pair-score ${pair.can_connect ? 'score-connect' : 'score-no-connect'}">
                            ${pair.can_connect ? '✓ CONNECT' : '✗ NO CONNECT'} (${pair.score.toFixed(0)}%)
                        </span>
                    </div>
                    <img src="data:image/png;base64,${img}" alt="Connection ${idx + 1}">
                    ${pair.message ? `<p class="pair-message">${pair.message}</p>` : ''}
                </div>
            `;
        }).join('');
    } else {
        assemblyImagesContainer.innerHTML = '<p class="no-connections">No matching arc segments found</p>';
    }
    
    document.getElementById('puzzleTotalScore').textContent = data.scores.total + '%';
    document.getElementById('puzzleFitScore').textContent = data.scores.puzzle_fit + '%';
    document.getElementById('puzzleFitBar').style.width = data.scores.puzzle_fit + '%';
    
    const scoreCircle = document.getElementById('puzzleScoreCircle');
    scoreCircle.classList.remove('score-high', 'score-medium', 'score-low');
    if (data.scores.total >= 60) {
        scoreCircle.classList.add('score-high');
    } else if (data.scores.total >= 40) {
        scoreCircle.classList.add('score-medium');
    } else {
        scoreCircle.classList.add('score-low');
    }
    
    document.getElementById('puzzleVerdict').textContent = data.verdict;
    
    const connectionStatus = document.getElementById('connectionStatus');
    
    if (data.scores.total >= 40) {
        connectionStatus.innerHTML = `<span class="status-success"><i class="fas fa-check-circle"></i> These artifacts can potentially connect!</span>`;
        connectionStatus.className = 'connection-status status-success';
    } else {
        let failMsg = '<span class="status-fail"><i class="fas fa-times-circle"></i> ';
        if (data.disrupting_pieces && data.disrupting_pieces.length > 0) {
            failMsg += `Piece(s) ${data.disrupting_pieces.map(p => '#' + p).join(', ')} disrupt the connection`;
        } else {
            failMsg += "These artifacts won't connect together";
        }
        failMsg += '</span>';
        connectionStatus.innerHTML = failMsg;
        connectionStatus.className = 'connection-status status-fail';
    }
    
    openModal('puzzleCompareModal');
}

function checkAuth() {
    fetch('/check_auth')
        .then(res => res.json())
        .then(data => {
            if (!data.authenticated) {
                showAuthHero();
            }
        })
        .catch(() => showAuthHero());
}

function showAuthHero() {
    const hero = document.querySelector('.hero-section');
    if (hero) {
        hero.classList.add('auth-hero');
    }
}

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    window.addEventListener('scroll', function() {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (scrollY >= sectionTop - 200) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').substring(1) === current) {
                link.classList.add('active');
            }
        });
    });
}

function initModals() {
    const modals = document.querySelectorAll('.modal');
    const modalCloses = document.querySelectorAll('.modal-close');
    
    modalCloses.forEach(close => {
        close.addEventListener('click', function() {
            const modalId = this.getAttribute('data-modal');
            closeModal(modalId);
        });
    });
    
    window.addEventListener('click', function(e) {
        modals.forEach(modal => {
            if (e.target === modal) {
                modal.classList.remove('active');
            }
        });
    });
    
    document.getElementById('loginBtn')?.addEventListener('click', () => openModal('loginModal'));
    document.getElementById('heroLoginBtn')?.addEventListener('click', () => openModal('loginModal'));
    document.getElementById('heroRegisterBtn')?.addEventListener('click', () => openModal('registerModal'));
    document.getElementById('switchToRegister')?.addEventListener('click', (e) => {
        e.preventDefault();
        closeModal('loginModal');
        setTimeout(() => openModal('registerModal'), 200);
    });
    document.getElementById('switchToLogin')?.addEventListener('click', (e) => {
        e.preventDefault();
        closeModal('registerModal');
        setTimeout(() => openModal('loginModal'), 200);
    });
    
    document.getElementById('logoutBtn')?.addEventListener('click', logout);
}

function openModal(modalId) {
    document.getElementById(modalId)?.classList.add('active');
}

function closeModal(modalId) {
    document.getElementById(modalId)?.classList.remove('active');
}

function initAuthForms() {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');
    
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(loginForm);
            const data = {
                username: formData.get('username'),
                password: formData.get('password')
            };
            
            showLoading('Logging in...');
            
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                hideLoading();
                
                if (result.error) {
                    showAuthError(loginForm, result.error);
                } else {
                    showAuthSuccess(loginForm, 'Login successful! Redirecting...');
                    setTimeout(() => {
                        location.reload();
                    }, 1000);
                }
            } catch (error) {
                hideLoading();
                showAuthError(loginForm, 'Login failed. Please try again.');
            }
        });
    }
    
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const password = document.getElementById('registerPassword').value;
            const confirmPassword = document.getElementById('registerConfirmPassword').value;
            
            if (password !== confirmPassword) {
                showAuthError(registerForm, 'Passwords do not match');
                return;
            }
            
            const formData = new FormData(registerForm);
            const data = {
                username: formData.get('username'),
                email: formData.get('email'),
                password: formData.get('password')
            };
            
            showLoading('Creating account...');
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                hideLoading();
                
                if (result.error) {
                    showAuthError(registerForm, result.error);
                } else {
                    showAuthSuccess(registerForm, 'Registration successful! Please login.');
                    setTimeout(() => {
                        closeModal('registerModal');
                        registerForm.reset();
                        openModal('loginModal');
                    }, 1500);
                }
            } catch (error) {
                hideLoading();
                showAuthError(registerForm, 'Registration failed. Please try again.');
            }
        });
    }
}

async function logout() {
    try {
        await fetch('/logout', { method: 'POST' });
        location.reload();
    } catch (error) {
        console.error('Logout failed:', error);
    }
}

function initUploadForms() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function() {
            const label = this.nextElementSibling;
            const fileName = this.files[0]?.name || 'Choose Image';
            label.querySelector('span').textContent = fileName;
            
            if (this.files[0]) {
                label.style.background = 'rgba(46, 125, 50, 0.3)';
                label.style.borderColor = 'var(--earth-forest)';
            }
        });
    });
    
    const addForm = document.getElementById('addToDatabaseForm');
    if (addForm) {
        addForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await handleAddToDatabase();
        });
    }
}

async function handleAddToDatabase() {
    const form = document.getElementById('addToDatabaseForm');
    const formData = new FormData(form);
    
    showLoading('Adding to collection...');
    
    try {
        const response = await fetch('/add_to_database', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        hideLoading();
        
        if (result.error) {
            showError(result.error, 'resultsContainer');
        } else {
            showSuccess(result.message, 'resultsContainer');
            form.reset();
            setTimeout(() => location.reload(), 1500);
        }
    } catch (error) {
        hideLoading();
        showError(`Upload failed: ${error.message}`, 'resultsContainer');
    }
}

function showComparisonResults(similarArtifacts, uploadedResultImage) {
    const resultsContainer = document.getElementById('resultsContainer');
    
    let html = '';
    
    if (uploadedResultImage) {
        html += `
            <div class="uploaded-result-section">
                <h3>Uploaded Image YOLO Result</h3>
                <img src="${uploadedResultImage}" alt="YOLO Result" class="uploaded-result-image">
            </div>
        `;
    }
    
    if (!similarArtifacts || similarArtifacts.length === 0) {
        html += `
            <div class="error-message">
                <i class="fas fa-exclamation-triangle"></i>
                <p>No similar artifacts found in your collection</p>
            </div>
        `;
    } else {
        html += '<h3 style="color: var(--earth-brown-primary); margin-bottom: 2rem; text-align: center; font-family: Playfair Display, serif;">Most Similar Artifacts</h3>';
        
        similarArtifacts.forEach((artifact, index) => {
            const similarityClass = getSimilarityClass(artifact.similarity_score);
            
            html += `
                <div class="similarity-card">
                    <div class="similarity-images">
                        <img src="${artifact.result_image || artifact.image_base64}" alt="${artifact.name}" class="similarity-image">
                    </div>
                    <div class="similarity-info">
                        <h4 class="similarity-name">${artifact.name}</h4>
                        <p style="color: var(--earth-brown); margin: 0.5rem 0;">
                            <i class="fas fa-calendar"></i> Added: ${artifact.timestamp}
                        </p>
                        <button class="earth-btn secondary view-similar-btn" data-id="${artifact.id}">
                            <i class="fas fa-eye"></i> View Details
                        </button>
                    </div>
                    <div class="similarity-score ${similarityClass}">
                        Similarity: ${artifact.similarity_score.toFixed(2)}%
                    </div>
                </div>
            `;
        });
    }
    
    resultsContainer.innerHTML = html;
    
    document.querySelectorAll('.view-similar-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const artifactId = btn.getAttribute('data-id');
            viewArtifactDetail(artifactId);
        });
    });
}

function getSimilarityClass(score) {
    if (score >= 50) return 'high-similarity';
    if (score >= 20) return 'medium-similarity';
    return 'low-similarity';
}

function initSearch() {
    const searchInput = document.getElementById('searchInput');
    const eraFilter = document.getElementById('eraFilter');
    const originFilter = document.getElementById('originFilter');
    const artifactCards = document.querySelectorAll('.artifact-card');
    
    const eras = new Set();
    const origins = new Set();
    
    artifactCards.forEach(card => {
        const era = card.getAttribute('data-era');
        const origin = card.getAttribute('data-origin');
        if (era) eras.add(era);
        if (origin) origins.add(origin);
    });
    
    eras.forEach(era => {
        const option = document.createElement('option');
        option.value = era;
        option.textContent = era;
        eraFilter?.appendChild(option);
    });
    
    origins.forEach(origin => {
        const option = document.createElement('option');
        option.value = origin;
        option.textContent = origin;
        originFilter?.appendChild(option);
    });
    
    function filterArtifacts() {
        const searchTerm = searchInput?.value.toLowerCase() || '';
        const eraValue = eraFilter?.value.toLowerCase() || '';
        const originValue = originFilter?.value.toLowerCase() || '';
        
        artifactCards.forEach(card => {
            const name = card.getAttribute('data-name') || '';
            const era = card.getAttribute('data-era') || '';
            const origin = card.getAttribute('data-origin') || '';
            
            const matchesSearch = name.includes(searchTerm);
            const matchesEra = !eraValue || era === eraValue;
            const matchesOrigin = !originValue || origin === originValue;
            
            if (matchesSearch && matchesEra && matchesOrigin) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
    }
    
    searchInput?.addEventListener('input', filterArtifacts);
    eraFilter?.addEventListener('change', filterArtifacts);
    originFilter?.addEventListener('change', filterArtifacts);
}

function initDatabaseControls() {
    const clearButton = document.getElementById('clearDatabase');
    
    if (clearButton) {
        clearButton.addEventListener('click', async function() {
            if (confirm('Are you sure you want to clear all artifacts? This cannot be undone.')) {
                const password = prompt('Enter your password to confirm:');
                
                if (!password) return;
                
                showLoading('Clearing collection...');
                
                try {
                    const response = await fetch('/clear_database', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ password })
                    });
                    
                    const result = await response.json();
                    hideLoading();
                    
                    if (result.error) {
                        showError(result.error, 'resultsContainer');
                    } else {
                        showSuccess(result.message, 'resultsContainer');
                        setTimeout(() => location.reload(), 1500);
                    }
                } catch (error) {
                    hideLoading();
                    showError(`Failed to clear: ${error.message}`, 'resultsContainer');
                }
            }
        });
    }
}

function initArtifactDetails() {
    document.querySelectorAll('.view-btn, .view-detail-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const artifactId = btn.getAttribute('data-id');
            viewArtifactDetail(artifactId);
        });
    });
    
    document.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const artifactId = btn.getAttribute('data-id');
            if (confirm('Are you sure you want to delete this artifact?')) {
                await deleteArtifact(artifactId);
            }
        });
    });
    
    document.getElementById('detailDeleteBtn')?.addEventListener('click', () => {
        const artifactId = document.getElementById('detailDeleteBtn').getAttribute('data-id');
        if (confirm('Are you sure you want to delete this artifact?')) {
            closeModal('artifactDetailModal');
            deleteArtifact(artifactId);
        }
    });
}

let currentArtifactId = null;
let similarOffset = 0;

async function viewArtifactDetail(artifactId) {
    currentArtifactId = artifactId;
    similarOffset = 0;
    showLoading('Loading artifact details...');
    
    try {
        const response = await fetch(`/artifact/${artifactId}`);
        const artifact = await response.json();
        hideLoading();
        
        if (artifact.error) {
            showError(artifact.error, 'resultsContainer');
            return;
        }
        
        document.getElementById('detailImage').src = artifact.image_base64;
        document.getElementById('detailResultImage').src = artifact.result_image || artifact.image_base64;
        document.getElementById('detailName').textContent = artifact.name;
        document.getElementById('detailTimestamp').textContent = artifact.timestamp;
        document.getElementById('detailOrigin').textContent = artifact.origin || 'Unknown';
        document.getElementById('detailEra').textContent = artifact.era || 'Unknown';
        document.getElementById('detailDeleteBtn').setAttribute('data-id', artifact.id);
        
        const featuresGrid = document.getElementById('featuresGrid');
        
        if (artifact.features) {
            const features = [
                { label: 'Area', value: artifact.features.area?.toFixed(2), unit: 'cm²' },
                { label: 'Perimeter', value: artifact.features.perimeter?.toFixed(2), unit: 'cm' },
                { label: 'Width', value: artifact.features.width?.toFixed(2), unit: 'cm' },
                { label: 'Height', value: artifact.features.height?.toFixed(2), unit: 'cm' },
                { label: 'Aspect Ratio', value: artifact.features.aspect_ratio?.toFixed(2), unit: '' },
                { label: 'Mean R', value: artifact.features.mean_r?.toFixed(0), unit: '' },
                { label: 'Mean G', value: artifact.features.mean_g?.toFixed(0), unit: '' },
                { label: 'Mean B', value: artifact.features.mean_b?.toFixed(0), unit: '' }
            ];
            
            featuresGrid.innerHTML = features.map(f => `
                <div class="feature-item">
                    <span class="feature-label">${f.label}</span>
                    <span class="feature-value">${f.value ?? 'N/A'} ${f.unit}</span>
                </div>
            `).join('');
            
            document.getElementById('detailFeatures').style.display = 'block';
        } else {
            document.getElementById('detailFeatures').style.display = 'none';
        }
        
        displaySimilarArtifacts(artifact.similar_artifacts || [], artifact.all_similar_count || 0);
        
        openModal('artifactDetailModal');
        
    } catch (error) {
        hideLoading();
        showError(`Failed to load artifact: ${error.message}`, 'resultsContainer');
    }
}

function displaySimilarArtifacts(artifacts, totalCount) {
    const similarGrid = document.getElementById('similarGrid');
    const loadMoreBtn = document.getElementById('loadMoreSimilar');
    const similarSection = document.getElementById('similarSection');
    
    if (!artifacts || artifacts.length === 0) {
        similarSection.style.display = 'none';
        return;
    }
    
    similarSection.style.display = 'block';
    
    const html = artifacts.map(artifact => {
        const scoreClass = artifact.similarity_score >= 50 ? 'high-similarity' : 
                          artifact.similarity_score >= 20 ? 'medium-similarity' : 'low-similarity';
        return `
            <div class="similar-artifact-card">
                <img src="${artifact.result_image || artifact.image_base64}" alt="${artifact.name}" class="similar-artifact-image">
                <div class="similar-artifact-info">
                    <h5>${artifact.name}</h5>
                    <span class="similarity-badge ${scoreClass}">${artifact.similarity_score.toFixed(1)}%</span>
                </div>
                <button class="view-similar-btn" data-id="${artifact.id}">
                    <i class="fas fa-eye"></i>
                </button>
            </div>
        `;
    }).join('');
    
    similarGrid.innerHTML = html;
    
    similarOffset = artifacts.length;
    if (similarOffset < totalCount) {
        loadMoreBtn.style.display = 'inline-block';
        loadMoreBtn.textContent = `Load More (${totalCount - similarOffset} remaining)`;
    } else {
        loadMoreBtn.style.display = 'none';
    }
    
    document.querySelectorAll('.view-similar-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const id = btn.getAttribute('data-id');
            closeModal('artifactDetailModal');
            setTimeout(() => viewArtifactDetail(id), 300);
        });
    });
    
    loadMoreBtn.onclick = async () => {
        try {
            const response = await fetch(`/artifact/${currentArtifactId}/similar?offset=${similarOffset}&limit=5`);
            const data = await response.json();
            
            const currentHtml = similarGrid.innerHTML;
            const newHtml = data.artifacts.map(artifact => {
                const scoreClass = artifact.similarity_score >= 50 ? 'high-similarity' : 
                                  artifact.similarity_score >= 20 ? 'medium-similarity' : 'low-similarity';
                return `
                    <div class="similar-artifact-card">
                        <img src="${artifact.result_image || artifact.image_base64}" alt="${artifact.name}" class="similar-artifact-image">
                        <div class="similar-artifact-info">
                            <h5>${artifact.name}</h5>
                            <span class="similarity-badge ${scoreClass}">${artifact.similarity_score.toFixed(1)}%</span>
                        </div>
                        <button class="view-similar-btn" data-id="${artifact.id}">
                            <i class="fas fa-eye"></i>
                        </button>
                    </div>
                `;
            }).join('');
            
            similarGrid.innerHTML = currentHtml + newHtml;
            
            similarOffset += data.artifacts.length;
            if (!data.has_more) {
                loadMoreBtn.style.display = 'none';
            } else {
                loadMoreBtn.textContent = `Load More (${data.total - similarOffset} remaining)`;
            }
            
            document.querySelectorAll('.view-similar-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const id = btn.getAttribute('data-id');
                    closeModal('artifactDetailModal');
                    setTimeout(() => viewArtifactDetail(id), 300);
                });
            });
        } catch (error) {
            console.error('Failed to load more:', error);
        }
    };
}

async function deleteArtifact(artifactId) {
    showLoading('Deleting artifact...');
    
    try {
        const response = await fetch(`/artifact/${artifactId}`, { method: 'DELETE' });
        const result = await response.json();
        hideLoading();
        
        if (result.error) {
            showError(result.error, 'resultsContainer');
        } else {
            showSuccess(result.message, 'resultsContainer');
            setTimeout(() => location.reload(), 1000);
        }
    } catch (error) {
        hideLoading();
        showError(`Failed to delete: ${error.message}`, 'resultsContainer');
    }
}

function showLoading(message = 'Processing...') {
    const overlay = document.getElementById('loadingOverlay');
    overlay.querySelector('p').textContent = message;
    overlay.classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

function showError(message, containerId = 'resultsContainer') {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="error-message">
            <i class="fas fa-exclamation-triangle"></i>
            <p>${message}</p>
        </div>
    `;
    container.style.display = 'block';
}

function showAuthError(form, message) {
    const existingError = form.querySelector('.auth-error');
    if (existingError) existingError.remove();
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'auth-error';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
    form.insertBefore(errorDiv, form.firstChild);
}

function showAuthSuccess(form, message) {
    const existingMsg = form.querySelector('.auth-success');
    if (existingMsg) existingMsg.remove();
    
    const msgDiv = document.createElement('div');
    msgDiv.className = 'auth-success';
    msgDiv.innerHTML = `<i class="fas fa-check-circle"></i> ${message}`;
    form.insertBefore(msgDiv, form.firstChild);
}

function showSuccess(message, containerId = 'resultsContainer') {
    const container = document.getElementById(containerId);
    container.innerHTML = `
        <div class="success-message">
            <i class="fas fa-check-circle"></i>
            <p>${message}</p>
        </div>
    `;
    container.style.display = 'block';
}

function initSmoothScrolling() {
}

const style = document.createElement('style');
style.textContent = `
    .high-similarity {
        background: linear-gradient(135deg, #2e7d32, #388e3c) !important;
    }
    
    .medium-similarity {
        background: linear-gradient(135deg, #f57c00, #ef6c00) !important;
    }
    
    .low-similarity {
        background: linear-gradient(135deg, #6c757d, #495057) !important;
    }
    
    .nav-link.active {
        color: var(--earth-clay) !important;
        font-weight: 600;
    }
    
    .similarity-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        margin-bottom: 1rem;
    }
    
    .similarity-card .similarity-image {
        width: 80px;
        height: 80px;
        object-fit: cover;
        border-radius: 10px;
    }
    
    .similarity-card .similarity-info {
        flex: 1;
    }
    
    .similarity-card .similarity-score {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        white-space: nowrap;
    }
    
    .method-description {
        margin-top: 0.5rem;
        color: var(--earth-brown);
    }
    
    .similar-artifacts-section {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 2px solid var(--earth-sand);
    }
    
    .similar-artifacts-section h4 {
        color: var(--earth-brown-primary);
        margin-bottom: 1rem;
        font-family: 'Playfair Display', serif;
    }
    
    .similar-artifacts-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 1rem;
    }
    
    .similar-artifact-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .similar-artifact-card:hover {
        transform: translateY(-3px);
    }
    
    .similar-artifact-image {
        width: 100%;
        height: 100px;
        object-fit: cover;
    }
    
    .similar-artifact-info {
        padding: 0.5rem;
        text-align: center;
    }
    
    .similar-artifact-info h5 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        color: var(--earth-brown-primary);
    }
    
    .similarity-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        color: white;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .view-similar-btn {
        width: 100%;
        padding: 0.5rem;
        background: var(--earth-clay);
        color: white;
        border: none;
        cursor: pointer;
        transition: background 0.2s;
    }
    
    .view-similar-btn:hover {
        background: var(--earth-brown-primary);
    }
    
    /* Artifact Selection */
    .artifact-controls {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .selection-info {
        font-weight: 600;
        color: var(--earth-brown-primary);
    }
    
    .artifact-checkbox {
        position: absolute;
        top: 10px;
        left: 10px;
        z-index: 10;
    }
    
    .artifact-checkbox input {
        width: 20px;
        height: 20px;
        cursor: pointer;
    }
    
    .artifact-card.selected-for-compare {
        border: 3px solid var(--earth-clay);
        box-shadow: 0 0 20px rgba(139, 69, 19, 0.4);
    }
    
    /* Puzzle Comparison Modal */
    .puzzle-modal-header {
        text-align: center;
        padding: 1.5rem;
        border-bottom: 2px solid var(--earth-sand);
    }
    
    .puzzle-modal-header h2 {
        color: var(--earth-brown-primary);
        font-family: 'Playfair Display', serif;
    }
    
    .puzzle-modal-header h2 i {
        color: var(--earth-clay);
        margin-right: 0.5rem;
    }
    
    .puzzle-compare-section {
        padding: 1.5rem;
    }
    
    .puzzle-artifacts {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .puzzle-artifact-card {
        flex: 1;
        max-width: 200px;
        text-align: center;
    }
    
    .puzzle-artifact-card img {
        width: 100%;
        height: 150px;
        object-fit: cover;
        border-radius: 10px;
        border: 3px solid var(--earth-sand);
    }
    
    .puzzle-artifact-card h4 {
        margin-top: 0.5rem;
        color: var(--earth-brown-primary);
        font-size: 0.9rem;
    }
    
    .puzzle-vs {
        font-size: 2rem;
        color: var(--earth-clay);
    }
    
    .puzzle-results {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
    }
    
    .puzzle-score-section {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .puzzle-score-section h3 {
        color: var(--earth-brown-primary);
        margin-bottom: 1rem;
        font-family: 'Playfair Display', serif;
    }
    
    .puzzle-score-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        background: var(--earth-sand);
        border: 4px solid var(--earth-clay);
    }
    
    .puzzle-score-circle span {
        font-size: 1.8rem;
        font-weight: bold;
        color: var(--earth-brown-primary);
    }
    
    .puzzle-score-circle.score-high {
        background: linear-gradient(135deg, #2e7d32, #388e3c);
        border-color: #1b5e20;
    }
    
    .puzzle-score-circle.score-high span {
        color: white;
    }
    
    .puzzle-score-circle.score-medium {
        background: linear-gradient(135deg, #f57c00, #ef6c00);
        border-color: #e65100;
    }
    
    .puzzle-score-circle.score-medium span {
        color: white;
    }
    
    .puzzle-score-circle.score-low {
        background: linear-gradient(135deg, #6c757d, #495057);
        border-color: #343a40;
    }
    
    .puzzle-score-circle.score-low span {
        color: white;
    }
    
    .puzzle-breakdown {
        margin-bottom: 1.5rem;
    }
    
    .puzzle-metric {
        margin-bottom: 1rem;
    }
    
    .metric-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.3rem;
    }
    
    .metric-header i {
        color: var(--earth-clay);
    }
    
    .metric-header span:first-of-type {
        flex: 1;
        font-weight: 500;
        color: var(--earth-brown-primary);
    }
    
    .metric-value {
        font-weight: bold;
        color: var(--earth-brown-primary);
    }
    
    .metric-bar {
        height: 8px;
        background: var(--earth-sand);
        border-radius: 4px;
        overflow: hidden;
    }
    
    .metric-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--earth-clay), var(--earth-brown-primary));
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .puzzle-verdict {
        text-align: center;
        padding: 1rem;
        background: var(--earth-sand);
        border-radius: 10px;
        font-size: 1.2rem;
        font-weight: bold;
        color: var(--earth-brown-primary);
    }
    
    .puzzle-artifacts-grid {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    
    .puzzle-artifact-item {
        position: relative;
        text-align: center;
    }
    
    .puzzle-artifact-item img {
        width: 100px;
        height: 100px;
        object-fit: cover;
        border-radius: 10px;
        border: 3px solid var(--earth-clay);
    }
    
    .puzzle-artifact-item .artifact-num {
        position: absolute;
        top: -10px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--earth-clay);
        color: white;
        width: 25px;
        height: 25px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.8rem;
    }
    
    .puzzle-artifact-item p {
        margin: 0.5rem 0 0;
        font-size: 0.8rem;
        color: var(--earth-brown-primary);
    }
    
    .connection-diagram {
        margin-top: 1.5rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
    }
    
    .connection-diagram h4 {
        text-align: center;
        color: var(--earth-brown-primary);
        margin-bottom: 1rem;
    }
    
    .connection-canvas {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    
    .connection-pair {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: var(--earth-sand);
        border-radius: 20px;
    }
    
    .artifact-ref {
        font-weight: bold;
        color: var(--earth-clay);
    }
    
    .connection-icon {
        font-size: 1.2rem;
    }
    
    .connection-icon.can-connect {
        color: #2e7d32;
    }
    
    .connection-icon.cannot-connect {
        color: #c62828;
    }
    
    .pair-score {
        font-size: 0.8rem;
        color: var(--earth-brown-primary);
    }
    
    .connection-status {
        text-align: center;
        padding: 0.8rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .connection-status.status-success {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .connection-status.status-fail {
        background: #ffebee;
        color: #c62828;
    }
    
    .no-connection {
        font-size: 3rem;
        color: #c62828;
    }
    
    .puzzle-artifact-item.disrupting img {
        border-color: #c62828 !important;
        box-shadow: 0 0 15px rgba(198, 40, 40, 0.5);
    }
    
    .disrupt-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        background: #c62828;
        color: white;
        width: 25px;
        height: 25px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
    }
    
    .disrupt-label {
        color: #c62828 !important;
        font-weight: bold !important;
        font-size: 0.75rem !important;
    }
    
    .connection-arrows {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        width: 100%;
        margin-top: 1rem;
    }
    
    .connection-arrow {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .connection-arrow.can-connect {
        background: #e8f5e9;
    }
    
    .connection-arrow.cannot-connect {
        background: #ffebee;
    }
    
    .connection-arrow i {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    
    .connection-arrow.can-connect i {
        color: #2e7d32;
    }
    
    .connection-arrow.cannot-connect i {
        color: #c62828;
    }
    
    .connection-arrow span {
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .artifact-img-container {
        position: relative;
    }
    
    .connection-marker {
        position: absolute;
        z-index: 10;
        pointer-events: none;
    }
    
    .marker-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        background: #2e7d32;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: bold;
        white-space: nowrap;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .connection-marker.left .marker-icon {
        background: #1565c0;
    }
    
    .connection-marker.right .marker-icon {
        background: #2e7d32;
    }
    
    .connection-marker.top .marker-icon,
    .connection-marker.bottom .marker-icon {
        background: #7b1fa2;
    }
    
    .puzzle-artifact-item.puzzle-artifact-item {
        position: relative;
    }
    
    .puzzle-artifact-item.disrupting .artifact-img-container img {
        border-color: #c62828 !important;
    }
    
    .puzzle-artifacts-row {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
`;
document.head.appendChild(style);

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
        });
    }
});
