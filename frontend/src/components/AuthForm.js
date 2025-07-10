import React, { useState } from 'react';
import { 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword,
  signInWithPopup,
  GoogleAuthProvider,
  updateProfile
} from 'firebase/auth';
import { auth } from '../services/firebaseConfig';

const AuthForm = ({ onAuthSuccess }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    displayName: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      let userCredential;
      if (isLogin) {
        userCredential = await signInWithEmailAndPassword(auth, formData.email, formData.password);
      } else {
        userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
        if (formData.displayName) {
          await updateProfile(userCredential.user, {
            displayName: formData.displayName
          });
        }
      }
      
      // Call success callback with user data
      if (onAuthSuccess && userCredential.user) {
        onAuthSuccess(userCredential.user);
      }
    } catch (error) {
      let errorMessage = 'Terjadi kesalahan, silakan coba lagi.';
      
      switch (error.code) {
        case 'auth/user-not-found':
          errorMessage = 'Email tidak ditemukan.';
          break;
        case 'auth/wrong-password':
          errorMessage = 'Password salah.';
          break;
        case 'auth/email-already-in-use':
          errorMessage = 'Email sudah terdaftar.';
          break;
        case 'auth/weak-password':
          errorMessage = 'Password terlalu lemah. Minimal 6 karakter.';
          break;
        case 'auth/invalid-email':
          errorMessage = 'Format email tidak valid.';
          break;
        case 'auth/too-many-requests':
          errorMessage = 'Terlalu banyak percobaan. Silakan coba lagi nanti.';
          break;
        default:
          errorMessage = error.message;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setLoading(true);
    setError('');
    
    try {
      const provider = new GoogleAuthProvider();
      const userCredential = await signInWithPopup(auth, provider);
      
      // Call success callback with user data
      if (onAuthSuccess && userCredential.user) {
        onAuthSuccess(userCredential.user);
      }
    } catch (error) {
      let errorMessage = 'Gagal masuk dengan Google.';
      
      if (error.code === 'auth/popup-closed-by-user') {
        errorMessage = 'Popup ditutup. Silakan coba lagi.';
      } else if (error.code === 'auth/network-request-failed') {
        errorMessage = 'Periksa koneksi internet Anda.';
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-form">
        <div className="auth-header">
          <h2>{isLogin ? 'Masuk ke XGFitness' : 'Daftar XGFitness'}</h2>
          <p>
            {isLogin 
              ? 'Masuk untuk melanjutkan perjalanan fitness Anda' 
              : 'Bergabunglah dengan komunitas fitness terbesar Indonesia'
            }
          </p>
        </div>

        {error && (
          <div className="error-banner">
            <h3>Oops! Ada masalah</h3>
            <p>{error}</p>
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <div className="form-group">
              <label htmlFor="displayName">Nama Lengkap</label>
              <input
                type="text"
                id="displayName"
                name="displayName"
                value={formData.displayName}
                onChange={handleInputChange}
                placeholder="Masukkan nama lengkap Anda"
                required={!isLogin}
              />
            </div>
          )}

          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              name="email"
              value={formData.email}
              onChange={handleInputChange}
              placeholder="nama@email.com"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              placeholder="Masukkan password"
              required
              minLength={6}
            />
            {!isLogin && (
              <small className="form-hint">
                Password minimal 6 karakter
              </small>
            )}
          </div>

          <button 
            type="submit" 
            className="btn-primary submit-btn"
            disabled={loading}
          >
            {loading ? (
              <>
                <div className="spinner small"></div>
                {isLogin ? 'Masuk...' : 'Mendaftar...'}
              </>
            ) : (
              isLogin ? 'Masuk' : 'Daftar'
            )}
          </button>
        </form>

        <div className="auth-divider">
          <span>atau</span>
        </div>

        <button 
          onClick={handleGoogleSignIn}
          className="btn-google"
          disabled={loading}
        >
          <svg width="20" height="20" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
          </svg>
          {loading ? 'Memproses...' : 'Lanjutkan dengan Google'}
        </button>

        <div className="auth-switch">
          <p>
            {isLogin ? 'Belum punya akun?' : 'Sudah punya akun?'}
            <button 
              type="button"
              onClick={() => {
                setIsLogin(!isLogin);
                setError('');
                setFormData({ email: '', password: '', displayName: '' });
              }}
              className="link-button"
            >
              {isLogin ? 'Daftar di sini' : 'Masuk di sini'}
            </button>
          </p>
        </div>

        <div className="auth-features">
          <h4>Mengapa bergabung dengan XGFitness?</h4>
          <ul>
            <li>🎯 Rekomendasi workout yang dipersonalisasi</li>
            <li>🍎 Panduan nutrisi lengkap dengan makanan Indonesia</li>
            <li>📊 Tracking progress yang mudah dan akurat</li>
            <li>🤖 AI yang terus belajar dari kebiasaan Anda</li>
            <li>🏆 Pencapaian dan motivasi harian</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AuthForm;